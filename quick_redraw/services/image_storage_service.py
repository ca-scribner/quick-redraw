import os
import cv2
from typing import List, Tuple
import pandas as pd
import numpy as np

from quick_redraw.data.image_record import ImageRecord
from quick_redraw.data.db_session import create_session, global_init, add_commit_close
from quick_redraw.data.training_data import TrainingData
from quick_redraw.services.metadata_service import add_record_to_metadata, find_record_by_id, \
    find_records_with_label_normalized


def store_image(label=None, drawing=None, metadata_id=None, raw_storage_location=None,
                normalized_storage_location=None) -> ImageRecord:
    """
    TODO: this docstring

    Note: Assumes metadata_db is initialized already
    Future: This feels too multi-purpose - is there a better way to refactor?
            Break into store_raw and store_normalized, that way I can enforce things like different dtypes (raw as uint)

    Args:
        label:
        drawing:
        metadata_id:
        raw_storage_location:
        normalized_storage_location:

    Returns:

    """
    if not _xor(raw_storage_location, normalized_storage_location):
        raise ValueError("Exactly one of raw_storage_location and normalized_storage_location must be specified")
    # Basic validation
    drawing = np.asarray(drawing)

    if not metadata_id:
        if not label:
            raise ValueError("Storing a new record requires a label")
        # put record into metadata_db
        m = add_record_to_metadata()
    else:
        # Get existing record
        m = find_record_by_id(metadata_id=metadata_id)

    # Update the metadata record and recommit
    if label:
        m.label = label

    # Store img file in raw_storage_location as label_metadataId (so it is unique and easily understood)
    # TODO: Handle GCP
    if raw_storage_location:
        filepath = os.path.join(raw_storage_location, f"{m.label}_{m.id}.png")
        m.file_raw = filepath
    elif normalized_storage_location:
        filepath = os.path.join(normalized_storage_location, f"{m.label}_{m.id}.png")
        m.file_normalized = filepath
    else:
        raise ValueError("No storage location specified -- how did I get here? Should have gotten caught by input "
                         "validation.")
    cv2.imwrite(filepath, drawing)

    # TODO: If unsuccessful, remove from metadata_db

    # Store location in metadata as well
    add_commit_close(m, expire_on_commit=False)
    return m


def _xor(raw_storage_location, normalized_storage_location):
    return bool(raw_storage_location) != bool(normalized_storage_location)


def load_image_from_id(metadata_id: int, storage_location: str = "normalized", return_record=False):
    """
    Loads and returns a drawing given a metadata id and storage_location.  Optionally returns record as well

    Args:
        metadata_id (str): Drawing metadata id
        storage_location (str): Any storage location accepted by load_drawing_from_record
        return_record (bool): If True, returns (drawing, Metadata).  Else, returns drawing.

    Returns:
        If return_record:
            (drawing, Metadata), where drawing is a numpy array
        else:
            drawing, where drawing is a numpy array
    """
    record = find_record_by_id(metadata_id)
    drawing = load_image_from_record(record, storage_location=storage_location)
    if return_record:
        return drawing, record
    else:
        return drawing


def load_image_from_record(record: ImageRecord, storage_location: str = "normalized") -> np.array:
    """
    Loads and returns a drawing given a Metadata object and storage_location

    Args:
        record (ImageRecord): Drawing metadata object
        storage_location (str): Any of "normalized" or "raw", denoting the type of file to be returned

    Returns:
        (np.array): drawing in numpy array format
    """
    valid_storage_locations = ['normalized', 'raw']
    if storage_location not in valid_storage_locations:
        raise ValueError(f"Invalid storage location '{storage_location}, must be one of "
                         f"'{', '.join(valid_storage_locations)}'")
    else:
        storage_location = "file_" + storage_location

    # Future: If storage_location does not exist, this wont give a very helpful error
    return cv2.imread(getattr(record, storage_location))


def load_normalized_images(label: str = None, return_records: bool = False) \
        -> List[Tuple[str, np.array]]:
    records = find_records_with_label_normalized(label)

    # Future: If storage_location does not exist, this wont give a very helpful error
    # Validate storage_location here and remove bad records?  Or validate at use?
    images = [(record.label, load_image_from_record(record, storage_location="normalized")) for record in records]

    if return_records:
        return images, records
    else:
        return images


def load_training_data_to_dataframe(training_data_id=None):
    """
    Returns train and test images and labels for a TrainingData entry

    By default returns the most recent TrainingData entry.

    Args:
        training_data_id (TrainingData.id): (OPTIONAL) If specified, returns TrainingData with this id.  Otherwise returns
                                         the most recently created TrainingData

    Returns:
        Tuple of:
            (list): Training ImageRecords (eg: x_train)
            (list): Testing ImageRecords (eg: x_test)
            (list): Training labels (eg: y_train) encoded to integers
            (list): Testing labels (eg: y_test) encoded to integers
            (list): List of labels indexed the same as y_train/y_test
    """
    s = create_session()
    if training_data_id:
        td: TrainingData = s.query(TrainingData).filter(TrainingData.id == training_data_id).first()
    else:
        td: TrainingData = s.query(TrainingData).order_by(TrainingData.created_date.desc()).first()

    # Build pd.DataFrame objects that are suitable for tf.keras.preprocessing.image.ImageDataGenerator
    df_train = _image_records_to_dataframe(td.training_images, td.label_to_index)
    df_test = _image_records_to_dataframe(td.testing_images, td.label_to_index)

    return df_train, df_test, td.index_to_label


def _image_records_to_dataframe(image_records, label_to_index):
    data = {
        'filename': [img.normalized for img in image_records],
        'class': [label_to_index[img.label] for img in image_records],
        'class_string': [img.label for img in image_records],
        'image_id': [img.id for img in image_records],
    }

    return pd.DataFrame(data)
