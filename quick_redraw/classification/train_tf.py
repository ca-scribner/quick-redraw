import datetime
import inspect
import os
import argparse
from typing import Tuple

import ray
from mlflow.tracking import MlflowClient
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS, MLFLowLogger

from quick_redraw.data.db_session import global_init
from quick_redraw.services.image_storage_service import load_training_data_to_dataframe


# Based off of https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tf_mnist_example.py and
# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_tune_cifar10_with_keras.py


class MyTrainable(tune.Trainable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_model(self, output_shape):
        # FUTURE: Need a better way to do this
        # Inspect the model signature to identify which config parameters are for this model, then instantiate using
        # them
        # TODO: Change this so I'm passing the model class name to import, not the class itself in config
        accepted_keys = set(inspect.getfullargspec(self.config['model_class']).args)
        model_kwargs = {key: value for key, value in self.config.items()
                        if key in accepted_keys}

        self.model = self.config['model_class'](n_output=output_shape,
                                                **model_kwargs)

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

        self._build_model(len(self.index_to_label))

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
        train_history = self.model.fit(
            x=self.train_gen,
            epochs=self.config.get("epochs", 1)
        )

        # Could combine this above using the validation argument
        test_loss, test_accuracy = self.model.evaluate(x=self.test_gen,
                                                       # callback=...  # Use to debug later?
                                                       )

        return {
            "epoch": self.iteration,
            # steps within this training epoch
            "loss": train_history.history['loss'],
            "accuracy": train_history.history['accuracy'],
            "test_loss": float(test_loss),
            "mean_accuracy": float(test_accuracy),
        }

    # FUTURE: Implement _save and _restore to allow for checkpointing?


def parse_arguments() -> Tuple[str, int, str, str, bool]:
    parser = argparse.ArgumentParser(description="Performs a hyperparameter search for a model on a specified set of"
                                                 "training data")
    parser.add_argument('model', type=str, action="store",
                        help="Name of model to use during training.  Model must be implemented in models dir with a "
                             "filename that is the all-lowercase version of its modelname.  Eg: MyModel would be "
                             "implemented in quick_redraw.models.mymodel.MyModel.  quick_redraw.models.mymodel must "
                             "also provide quick_redraw.models.mymodel.SEARCH_PARAMS which defines the grid/random "
                             "search parameters for this model")
    parser.add_argument('td_id', type=int, action="store",
                        help="ID number of the TrainingData entry that defines Train/Test data")
    parser.add_argument('db_location', type=str, action="store",
                        help="Path to the database with TrainingDataRecord table")
    parser.add_argument('mlflow_uri', type=str, action="store",
                        help="Path to the mlflow tracking server (remote and local supported)")
    parser.add_argument('--smoke_test', action="store_true",
                        help="Limit to 3 epochs per training run for debugging")
    args = parser.parse_args()

    # FUTURE: For now, assume db_location is local and make it absolute relative to here.  This is needed because all
    #  tune training runs will be run in their own location.  For distributed training using ray I'll need a better
    #  solution (cloud db?)
    import os
    args.db_location = os.path.abspath(args.db_location)

    return args.model, args.td_id, args.db_location, args.mlflow_uri, args.smoke_test


def main(model_name: str, td_id: int, metadata_location: str, mlflow_uri: str, smoke_test: bool):
    """
    FUTURE: Docstring

    Args:
        model_name:
        td_id:
        metadata_location:
        mlflow_uri:
        smoke_test:

    Returns:

    """
    # # For debugging
    ray.init(local_mode=True, num_cpus=1)
    # Limit us to N CPUs (threads) during testing
    # ray.init(num_cpus=4)

    print("WARNING: model argument not fully implemented.")

    # Initialize the mlflow session and create an experiment
    client = MlflowClient(tracking_uri=mlflow_uri)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{now}_{model_name}_tdr.{td_id}"
    mlflow_experiment_id = client.create_experiment(name=experiment_name)

    if smoke_test:
        training_iteration = 3
    else:
        training_iteration = 10

    import importlib
    model_package = importlib.import_module(f".{model_name.lower()}", "quick_redraw.models")

    # Load hyperparameter configs from model, prepended with a parameter identifier

    config = model_package.SEARCH_PARAMS

    # Add other config data
    config["mlflow_experiment_id"] = mlflow_experiment_id
    config["training_data_id"] = td_id
    config["metadata_location"] = metadata_location
    config["model_class"] = getattr(model_package, model_name)

    analysis = tune.run(
        MyTrainable,
        stop={"training_iteration": training_iteration},
        verbose=1,
        config=config,
        # Are training results described here everything (full model checkpoint available?)  Guess I just need params
        # for a retrain on all data
        local_dir="./ray_results/",  # Where training results are stored locally
        # upload_dir="s3://...",  # Could use this to store training progress to remote storage
        # Loggers are called at the end of each training_iteration
        loggers=DEFAULT_LOGGERS + (MLFLowLogger, ),
    )

    print(f"Best configuration is: {analysis.get_best_config('mean_accuracy')}")


if __name__ == '__main__':
    # FUTURE: Add ray cluster address to CLI
    main(*parse_arguments())
