from typing import Optional

import sqlalchemy.exc

from quick_redraw.data.metadata_db_session import create_session
from quick_redraw.data.metadata import Metadata


def add_record_to_metadata(label: str, raw_storage_location: str = None,
                           normalized_storage_location: str = None) -> Metadata:
    m = Metadata()
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


def find_record_by_id(metadata_id: int) -> Metadata:
    s = create_session()
    m = s.query(Metadata).filter(Metadata.id == metadata_id).first()
    s.close()
    return m


def find_records_with_label(label):
    # Should this return only raw/norm'd?  or make separate funcs for those?
    raise NotImplementedError()
