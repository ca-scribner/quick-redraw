import os

import numpy as np
import argparse

import sklearn.model_selection

from quick_redraw.data.db_session import global_init, create_session
from quick_redraw.data.training_data import TrainingData
from quick_redraw.services.image_storage_service import load_normalized_images
from quick_redraw.services.metadata_service import find_records_with_label_normalized


def create_training_data_from_image_db(test_size=0.2, random_state=42):
    """
    Creates a train-test split of the current image db stored in a TrainingData table

    Args:
        test_size:
        random_state:

    Returns:

    """
    normalized_images = find_records_with_label_normalized()
    labels = [img.label for img in normalized_images]
    index_to_label = sorted(set(labels))
    label_to_index = {index_to_label[i]: i for i in range(len(index_to_label))}

    training_images, testing_images = sklearn.model_selection.train_test_split(normalized_images,
                                                                               test_size=test_size,
                                                                               stratify=labels,
                                                                               random_state=random_state,
                                                                               )

    td = TrainingData(label_to_index=label_to_index, index_to_label=index_to_label)
    td.training_images.extend(training_images)
    td.testing_images.extend(testing_images)

    s = create_session()
    s.add(td)
    s.commit()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect normalized images and split/store them in train and test "
                                                 "groups")
    parser.add_argument('db_location', type=str, action="store",
                        help="Path to the database")
    parser.add_argument('test_size', type=float, action="store",
                        help="Fractional size of the test database")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    global_init(args.db_location)

    create_training_data_from_image_db(test_size=args.test_size)
