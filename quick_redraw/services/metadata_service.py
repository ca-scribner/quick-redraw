from typing import Optional, List

import sqlalchemy.exc

from quick_redraw.data.db_session import create_session
from quick_redraw.data.image_record import ImageRecord


def add_record_to_metadata(label: str = None, raw_storage_location: str = None,
                           normalized_storage_location: str = None) -> ImageRecord:
    m = ImageRecord()
    m.label = label
    m.file_raw = raw_storage_location
    m.file_normalized = normalized_storage_location

    s = create_session()
    s.add(m)

    # commit data to DB but key an unexpired version to pass back to caller
    # This is essentially a copy of the state of m at the time of commit, but wont it wont automatically refresh it's
    # data if we interact with it (expire_on_commit=True tells m it needs to resync next time we use it)
    s.expire_on_commit = False
    s.commit()
    s.close()

    return m


def find_record_by_id(metadata_id: int) -> ImageRecord:
    s = create_session()
    m = s.query(ImageRecord).filter(ImageRecord.id == metadata_id).first()
    s.close()
    return m


def find_records_with_label_normalized(label: str = None) -> List[ImageRecord]:
    s = create_session()
    q = s.query(ImageRecord)
    if label:
        q = q.filter(ImageRecord.label == label)
    q = q.filter(ImageRecord.file_normalized.isnot(None))
    results = q.all()
    s.close()
    return results


def find_records_unnormalized() -> List[ImageRecord]:
    s = create_session()
    q = s.query(ImageRecord)\
        .filter(ImageRecord.file_normalized is not None)
    results = q.all()
    s.close()
    return results
