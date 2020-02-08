import argparse
import os
from typing import List, Tuple

import numpy as np

from quick_redraw.data.metadata import Metadata
from quick_redraw.data.metadata_db_session import create_session, global_init, add_commit_close
from quick_redraw.services.metadata_service import add_record_to_metadata, find_record_by_id


def store_image(label=None, drawing=None, metadata_id=None, raw_storage_location=None, normalized_storage_location=None):
    """
    TODO: this docstring

    Note: Assumes metadata_db is initialized already
    Future: This feels too multi-purpose - is there a better way to refactor?

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
    drawing = np.asarray(drawing, dtype=int)

    if not metadata_id:
        if not label:
            raise ValueError("Storing a new record requires a label")
        # put record into metadata_db
        m = add_record_to_metadata()
        print(f"Added metadata record {m}")
    else:
        # Get existing record
        m = find_record_by_id(metadata_id=metadata_id)

    # Update the metadata record and recommit
    if label:
        m.label = label

    # Store img file in raw_storage_location as label_metadataId (so it is unique and easily understood)
    # TODO: Handle GCP
    if raw_storage_location:
        filepath = os.path.join(raw_storage_location, f"{m.label}_{m.id}.npy")
        m.file_raw = filepath
    elif normalized_storage_location:
        filepath = os.path.join(normalized_storage_location, f"{m.label}_{m.id}.npy")
        m.file_normalized = filepath
    else:
        raise ValueError("No storage location specified -- how did I get here? Should have gotten caught by input "
                         "validation.")
    np.save(filepath, drawing)

    # TODO: If unsuccessful, remove from metadata_db

    # Store location in metadata as well
    add_commit_close(m)


def _xor(raw_storage_location, normalized_storage_location):
    print(f"raw_storage_location = {raw_storage_location}")
    print(f"normalized_storage_location = {normalized_storage_location}")
    print(
        f"bool(raw_storage_location) != bool(normalized_storage_location) = {bool(raw_storage_location) != bool(normalized_storage_location)}")
    return bool(raw_storage_location) != bool(normalized_storage_location)


def load_drawings(label: str = None, storage_location: str = 'normalized') -> List[Tuple[str, np.array]]:
    valid_storage_locations = ['normalized', 'raw']
    if storage_location not in valid_storage_locations:
        raise ValueError(f"Invalid storage location '{storage_location}, must be one of "
                         f"'{', '.join(valid_storage_locations)}'")
    else:
        storage_location = "file_" + storage_location
    s = create_session()
    if label:
        records = s.query(Metadata).filter(Metadata.label == label).all()
    else:
        records = s.query(Metadata).all()
    s.close()
    drawings = [(record.label, np.load(getattr(record, storage_location))) for record in records]
    return drawings


# # Queued for removal.  I think this is better living in load_data.  It is basically just code to enable testing,
# # so the testing part of this should just BE in testing.
# # Use this to make tests, but smaller tests (single file load, convert to drawing, check result)
# def load_image(filename, filetype):
#     if filetype == "single_npy":
#         return [load_image_npy(filename)]
#     elif filetype == "multiple_bin":
#         return load_images_bin(filename)
#     else:
#         raise ValueError(f"Unsupported filetype {filetype}")
#
#
# def load_image_npy(filename):
#     return np.load(filename)
#
#
# def load_images_bin(filename):
#     raise NotImplementedError
#
#
# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Store an image file in the metadata database and raw file storage "
#                                                  "location")
#     parser.add_argument('label', type=str, action="store",
#                         help="Label associated with the file")
#     parser.add_argument('filename', type=str, action="store",
#                         help="Full local filepath to the file to be loaded")
#     parser.add_argument('metadata_location', type=str, action="store",
#                         help="Path to metadata database (local or gcp bucket)")
#     parser.add_argument('--raw_storage_location', type=str, action="store",
#                         help="Location to store raw files (local or gcp bucket).  Exactly one of raw and normalized "
#                              "location can be specified")
#     parser.add_argument('--normalized_storage_location', type=str, action="store",
#                         help="Location to store normalized files (local or gcp bucket).  Exactly one of raw and "
#                              "normalized location can be specified")
#     parser.add_argument("--filetype", type=str, action="store", default="single_npy",
#                         help="Type of file to load image(s) from.  single_npy loads a single nxm image as an np.array "
#                              "from an npy file.  multiple_bin loads all images from a Quick, Draw! .bin format file")
#     args = parser.parse_args()
#     return args
#
#
# def init_db(metadata_location):
#     global_init(metadata_location)
#
#
# def main(label, filename, metadata_location, raw_storage_location, normalized_storage_location, filetype="single_npy"):
#     print(f"label = {label}")
#     print(f"filename = {filename}")
#     print(f"metadata_location = {metadata_location}")
#     print(f"raw_storage_location = {raw_storage_location}")
#     print(f"normalized_storage_location = {normalized_storage_location}")
#     print(f"filetype = {filetype}")
#
#     # If running standalone, need to init db
#     init_db(metadata_location)
#
#     drawings = load_image(filename, filetype)
#
#     for drawing in drawings:
#         print("loaded drawing:")
#         print(f"type(drawing) = {type(drawing)}")
#         print(f"drawing.shape = {drawing.shape}")
#         print(drawing)
#
#         store_image(label, drawing, raw_storage_location=raw_storage_location,
#                     normalized_storage_location=normalized_storage_location)
#
#
# if __name__ == '__main__':
#     args = parse_arguments()
#     label = args.label
#     filename = args.filename
#     metadata_location = args.metadata_location
#     raw_storage_location = args.raw_storage_location
#     normalized_storage_location = args.normalized_storage_location
#     filetype = args.filetype
#
#     main(label, filename, metadata_location, raw_storage_location, normalized_storage_location, filetype)
