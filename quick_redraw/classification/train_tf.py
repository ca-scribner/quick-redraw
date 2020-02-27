import os

import ray
from ray import tune

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from quick_redraw.data.db_session import global_init
from quick_redraw.services.image_storage_service import load_training_data_to_dataframe


# Based off of https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tf_mnist_example.py and
# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_tune_cifar10_with_keras.py


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


class MyTrainable(tune.Trainable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_model(self, output_shape):
        model = SimpleConv(n_output=output_shape,
                           conv_filters=self.config['conv_filters'],
                           conv_kernel_size=self.config['conv_kernel_size'],
                           conv_stride=self.config['conv_stride'],
                           pool_size=self.config['pool_size'],
                           dense_layer_size=self.config['dense_layer_size'],
                           dropout=self.config['dropout'],
                           )
        return model

    def _load_data(self, config):
        global_init(config['metadata_location'])
        df_train, df_test, index_to_label = load_training_data_to_dataframe()

        # Fixes issue with interaction between Ray actors and TF as described here:
        # https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tf_mnist_example.py
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        idg_train = ImageDataGenerator()
        flow_args = dict(
            x_col='filename',
            y_col='class_id_as_string',
            batch_size=config['batchsize'],
            class_mode='categorical',
            shuffle=True,
        )
        self.train_gen = idg_train.flow_from_dataframe(df_train,
                                                       # save_to_dir='./',  # For debugging...
                                                       **flow_args
                                                       )
        idg_test = ImageDataGenerator()
        self.test_gen = idg_test.flow_from_dataframe(df_test,
                                                     **flow_args
                                                     )

        self.index_to_label = index_to_label

    def _setup(self, config):
        # Note: config is also available from self.config

        # Fixes issue with interaction between Ray actors and TF as described here:
        # https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tf_mnist_example.py
        import tensorflow as tf

        # Apply defaults
        config['batchsize'] = config.get('batchsize', 32)

        self._load_data(config)

        self.model = self._build_model(len(self.index_to_label))

        # Don't really need these in self, but kept from a previous version
        # Use categoricalCrossentropy because ImageDataGenerator passes labels as n-d arrays
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(config['learning_rate'])

        self.model.compile(
            loss=self.loss_object,
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )

        # print("DOES THIS MODEL WORK?")
        # self.model.build((None, 28, 28, 1))
        # print(self.model.summary())

    def _train(self):
        history = self.model.fit(
            x=self.train_gen,
            epochs=self.config.get("epochs", 1)
        )

        # Could combine this above using the validation argument
        test_loss, test_accuracy = self.model.evaluate(x=self.test_gen,
                                                       # callback=...  # Use to debug later?
                                                       )

        return {
            "epoch": self.iteration,
            "loss": history.history['loss'],
            "accuracy": history.history['accuracy'],
            "test_loss": test_loss,
            "mean_accuracy": test_accuracy,
        }

    # FUTURE: Implement _save and _restore to allow for checkpointing?


if __name__ == '__main__':
    # FUTURE: Add CLI
    metadata_location = os.path.join(os.path.dirname(__file__), '..', '..', 'mock_data', 'untracked', 'metadata.sqlite')

    # For debugging
    # ray.init(local_mode=True)
    # Limit us to 2CPUs (threads) during testing
    ray.init(num_cpus=2)

    analysis = tune.run(
        MyTrainable,
        stop={"training_iteration": 3},
        verbose=1,
        config={
            "metadata_location": metadata_location,
            'conv_filters': tune.grid_search([32]),
            'conv_kernel_size': tune.grid_search([3]),
            'conv_stride': tune.grid_search([1]),
            'pool_size': tune.grid_search([2]),
            'dense_layer_size': tune.grid_search([64, 128]),
            'dropout': tune.grid_search([0.0, 0.2]),
            'learning_rate': tune.grid_search([0.0001])
            # 'epochs': tune.grid_search([1, 5, 10, 20])  # Training epochs.  Bother varying?  What would this affect?
            # 'dense_layer_size': tune.grid_search([64, 128]),
            # 'dropout': tune.grid_search([0.0, 0.2, 0.5]),
            # 'learning_rate': tune.grid_search([0.0001, 0.001, 0.01])
        },
        # Are training results described here everything (full model checkpoint available?)  Guess I just need params
        # for a retrain on all data
        local_dir="./ray_results/"  # Where training results are stored locally
        # upload_dir="s3://...",  # Could use this to store training progress to remote storage

    )

    print(f"Best configuration is: {analysis.get_best_config('mean_accuracy')}")

    fout = 'results.xlsx'
    print(f"Saving results to dataframe {fout}")
    analysis.dataframe().to_excel(fout)

    all_dataframes = analysis.trial_dataframes
    print(f"all_dataframes (this is a dataframe for each trial, with rows for each training iteration)= \n{all_dataframes}")

    print(f"analysis.trials = \n{analysis.trials}")

    import joblib
    joblib.dump(analysis, 'analysis.joblib')
