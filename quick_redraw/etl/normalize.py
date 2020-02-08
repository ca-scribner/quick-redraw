import cv2
import numpy as np

from quick_redraw.etl.save_load_images import store_image
from quick_redraw.services.metadata_service import find_record_by_id


def normalize_drawing_from_db(metadata_id: int, normalized_storage_location: str) -> None:
    """
    Pulls a raw image from storage, normalizes it, and stores it in normalized storage

    Args:
        metadata_id (int): Metadata id number
        normalized_storage_location (str): Path to location for storage of normalized images

    Returns:
        None
    """
    # Get m
    m = find_record_by_id(metadata_id)

    # Load npy from m
    drawing_raw = np.load(m.file_raw)

    drawing_normalized = normalize_drawing(drawing_raw)

    # Store image
    store_image(drawing=drawing_normalized, metadata_id=m.id, normalized_storage_location=normalized_storage_location)


def normalize_drawing(drawing: np.array):
    """
    Normalizes an image as numpy array to a fixed size, returning a new numpy array

    Args:
        drawing (np.array): Image as a numpy array

    Returns:
        (np.array)
    """
    # Resize
    normalized_drawing_size = (28, 28)
    drawing = cv2.resize(drawing, dsize=normalized_drawing_size)

    # Flatten
    drawing = drawing.reshape(-1)
    return drawing