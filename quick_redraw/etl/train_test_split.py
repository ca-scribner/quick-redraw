import os

import numpy as np
import argparse

import sklearn.model_selection

from quick_redraw.data.metadata_db_session import global_init, create_session
from quick_redraw.data.training_data import TrainingData
from quick_redraw.etl.save_load_images import load_drawings


def train_test_split(training_data_storage_location, test_size=0.2, random_state=42):
    # Fetch normalized images
    label_drawing_tuples, metadatas = load_drawings(storage_location='normalized', return_records=True)
    drawing_ids = [m.id for m in metadatas]

    labels, drawings = zip(*label_drawing_tuples)

    # print(f"len(labels) = {len(labels)}")
    # print(f"drawings.shape = {drawings.shape}")

    # Create int->class map
    class_names, labels_as_index = np.unique(labels, return_inverse=True)

    # train-test-split images
    x_train, x_test, y_train, y_test, id_train, id_test = sklearn.model_selection.train_test_split(
        drawings,
        labels_as_index,
        drawing_ids,
        test_size=test_size,
        random_state=random_state,
    )

    # Add training_data to db to get id
    td = TrainingData()
    print(f"td = {td}")
    s = create_session()
    s.expire_on_commit = False
    s.add(td)
    s.commit()
    print(f"td = {td}")

    try:
        # Store train/test data in files at storage location
        filepath_base = os.path.join(training_data_storage_location, str(td.id))
        np.save(filepath_base + "_train_x.npy", x_train)
        np.save(filepath_base + "_train_y.npy", y_train)
        np.save(filepath_base + "_test_x.npy", x_test)
        np.save(filepath_base + "_test_y.npy", y_test)

    except Exception as e:
        print("Caught exception - removing partial data from db")
        TrainingData.query.filter(TrainingData.id == td.id).delete()
        # Or do this?
        # s.delete(td)
        s.commit()
        raise e

    # Add metadata to training_data_metadata_path
    # (if fail, how do I remove the data from storage location?)
    td.class_names = class_names
    td.train = id_train
    td.test = id_test


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect normalized images and split/store them in train and test "
                                                 "groups")
    parser.add_argument('db_location', type=str, action="store",
                        help="Path to the database")
    parser.add_argument('training_data_storage_location', type=str, action="store",
                        help="Path to store train/test data")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    global_init(args.db_location)

    train_test_split(args.training_data_storage_location)
