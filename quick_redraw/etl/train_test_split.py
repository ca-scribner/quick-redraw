import argparse

from quick_redraw.data.metadata_db_session import global_init


def train_test_split(drawing_metadata_path, training_data_metadata_path, training_data_storage_location):
    # Fetch normalized images



    # Combine normalized images to single group

    # Create int->class map

    # train-test-split images

    # Store train/test data in files at storage location

    # Add metadata to training_data_metadata_path
    # (if fail, how do I remove the data from storage location)


    raise NotImplementedError()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect normalized images and split/store them in train and test "
                                                 "groups")
    parser.add_argument('drawing_metadata_path', type=str, action="store",
                        help="Path to the drawing metadata database")
    parser.add_argument('trainnig_metadata_path', type=str, action="store",
                        help="Path to the training data metadata database")
    parser.add_argument('training_data_storage_location', type=str, action="store",
                        help="Path to store train/test data")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    global_init(args.drawing_metadata_path)

    train_test_split()