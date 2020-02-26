import cv2
import numpy as np

from quick_redraw.services.image_storage_service import store_image, load_image_from_id
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
    drawing_raw = load_image_from_id(metadata_id, storage_location="raw")

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
    drawing = drawing.astype(np.float32)

    # Grayscale (am I getting the source color scheme right here?)
    # Going to grayscale changes this from a (NxMx3) to an (NxM) array
    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)

    # Resize
    normalized_drawing_size = (28, 28)
    drawing = cv2.resize(drawing, dsize=normalized_drawing_size)

    # # Flatten
    # drawing = drawing.reshape(-1)
    return drawing
