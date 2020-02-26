import os

import numpy as np

from ray import tune
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from quick_redraw.data.db_session import global_init, create_session
from quick_redraw.services.image_storage_service import load_normalized_images, load_training_data_to_dataframe


# Based off of https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tf_mnist_example.py


class SimpleConv(Model):
    def __init__(self, n_output, conv_filters=28, conv_kernel_size=None, conv_stride=1, pool_size=None,
                 dense_layer_size=128, dropout=0.0):
        if not conv_kernel_size:
            conv_kernel_size = (2, 2)

        super().__init__()
        self.layers_ = []
        self.layers_.append(Conv2D(filters=conv_filters,
                                  kernel_size=conv_kernel_size,
                                  strides=conv_stride))
        if pool_size:
            self.layers_.append(MaxPooling2D(pool_size=pool_size))
        self.layers_.append(Flatten())
        self.layers_.append(Dense(dense_layer_size, activation="relu"))
        self.layers_.append(Dropout(dropout))

        # Output layer
        self.layers_.append(Dense(n_output, activation='softmax'))

    def call(self, x):
        # TODO: super signature has training=None, mask=None.  Fix
        for layer in self.layers_:
            x = layer(x)
        return x


# DEBUG:
class MyModel(Model):
    def __init__(self, hiddens=128):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(hiddens, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class MyTrainable(tune.Trainable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_encoder_ = None

    def _setup(self, config):
        # Fixes issue with interaction between Ray actors and TF as described here:
        # https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tf_mnist_example.py
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Apply defaults
        config['batchsize'] = config.get('batchsize', 32)

        df_train, df_test, index_to_label = load_training_data_to_dataframe()

        idg_train = ImageDataGenerator()
        train_generator = idg_train.flow_from_dataframe(df_train, batch_size=config['batchsize'])
        idg_test = ImageDataGenerator()
        train_generator = idg_test.flow_from_dataframe(df_test, batch_size=config['batchsize'])

        #
        #
        #
        # x_train, y_train, x_test, y_test = load_data(config['metadata_location'])
        #
        # # Define encoder
        # # TODO: Where should I actually encode the label?  Should encoding be in ETL/db?")
        # label_encoder = LabelEncoder()
        # y_train = label_encoder.fit_transform(y_train)
        # y_test = label_encoder.transform(y_test)
        #
        # self.label_encoder_ = label_encoder
        #
        # # Infer number of output classes
        # print("NEED TO HANDLE NUMBER OF OUTPUTS AND MAPPING BETTER")
        # n_output = len(np.unique(y_train))
        #
        # # TODO: Do I need to add an extra channels dimension?  I'm using greyscale, but is it expected downstream?
        # x_train = x_train[..., tf.newaxis]
        # x_test = x_test[..., tf.newaxis]
        # # x_train = x_train[..., None]
        # # x_test = x_test[..., None]
        #
        #
        # self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        #     .shuffle(20000) \
        #     .batch(config['batchsize'])
        # self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        #     .batch(config['batchsize'])

        self.model = SimpleConv(n_output=len(index_to_label),
                                conv_filters=config['conv_filters'],
                                conv_kernel_size=config['conv_kernel_size'],
                                conv_stride=config['conv_stride'],
                                pool_size=config['pool_size'],
                                dense_layer_size=config['dense_layer_size'],
                                dropout=config['dropout'],
                                )
        # print("DOES THIS WORK?")
        # self.model.build((None, 28, 28, 1))
        # print(self.model.summary())

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(config['learning_rate'])
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(images)
                loss = self.loss_object(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_loss(loss)
            self.train_accuracy(labels, predictions)

        self.tf_train_step = train_step

        @tf.function
        def test_step(images, labels):
            predictions = self.model(images)
            test_loss = self.loss_object(labels, predictions)
            self.test_loss(test_loss)
            self.test_accuracy(labels, predictions)

        self.tf_test_step = test_step

    def _train(self):
        self._reset_metric_states()

        for i, (images, labels) in enumerate(self.train_ds):
            # Could insert main train iteration here
            self.tf_train_step(images, labels)

        for test_images, test_labels in self.test_ds:
            self.tf_test_step(test_images, test_labels)

        return {
            "epoch": self.iteration,
            "loss": self.train_loss.result().numpy(),
            "accuracy": self.train_accuracy.result().numpy(),
            "test_loss": self.test_loss.result().numpy(),
            "mean_accuracy": self.test_accuracy.result().numpy(),
        }

    # FUTURE: Implement _save and _restore to allow for checkpointing?

    def _reset_metric_states(self):
        """Reset the states of loss and accuracy metric containers"""
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()


def load_data(metadata_location):
    global_init(metadata_location)

    label_drawing_tuples = load_normalized_images(storage_location='normalized')
    if not label_drawing_tuples:
        raise ValueError("No training drawings found in db")
    labels, drawings = zip(*label_drawing_tuples)

    drawings = np.asarray(drawings)
    labels = np.asarray(labels)

    x_train, x_test, y_train, y_test = train_test_split(drawings, labels, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    metadata_location = os.path.join(os.path.dirname(__file__), '..', '..', 'mock_data', 'untracked', 'metadata.sqlite')
    print(f"metadata_location = {metadata_location}")
    global_init(metadata_location)
    s = create_session()
    if not s:
        raise ValueError("Uhho")

    # Where should we load/store data in code?  Can we load early so everyone gets it?  Or with workers on diff nodes
    # do we need to be lazy?  Can/should we bind data to MyTrainable or the Model etc?


    # mt = MyTrainable(config={"metadata_location": metadata_location,
    #                          "dense_layer_size": 32,
    #                          "conv_filters": 32,
    #                          "conv_kernel_size": 3,
    #                          "conv_stride": 1,
    #                          "pool_size": 2,
    #                          "dropout": 0.2,
    #                          "learning_rate": 0.001
    #                          })


    tune.run(
        MyTrainable,
        stop={"training_iteration": 50},
        verbose=1,
        config={
            "metadata_location": metadata_location,
            'conv_filters': tune.grid_search([32]),
            'conv_kernel_size': tune.grid_search([3]),
            'conv_stride': tune.grid_search([1]),
            'pool_size': tune.grid_search([2]),
            'dense_layer_size': tune.grid_search([64]),
            'dropout': tune.grid_search([0.0]),
            'learning_rate': tune.grid_search([0.0001])
            # 'dense_layer_size': tune.grid_search([64, 128]),
            # 'dropout': tune.grid_search([0.0, 0.2, 0.5]),
            # 'learning_rate': tune.grid_search([0.0001, 0.001, 0.01])
        }
    )