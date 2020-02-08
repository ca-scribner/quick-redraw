import cv2
import numpy as np

from quick_redraw.etl.save_load_images import store_image
from quick_redraw.services.metadata_service import find_record_by_id


def normalize_image(metadata_id, normalized_storage_location):
    # Get m
    m = find_record_by_id(metadata_id)

    # Load npy from m
    drawing_raw = np.load(m.file_raw)

    # resize npy
    normalized_drawing_shape = (28, 28)
    drawing_resized = cv2.resize(drawing_raw, dsize=normalized_drawing_shape)

    # Store image
    store_image(drawing=drawing_resized, metadata_id=m.id, normalized_storage_location=normalized_storage_location)
