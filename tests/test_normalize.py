import os
import tempfile
from unittest import mock

import pytest
import numpy as np

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quick_redraw.data.metadata import Metadata
from quick_redraw.data.metadata_db_session import global_init, create_session
from quick_redraw.data.modelbase import SqlAlchemyBase
from quick_redraw.etl.normalize import normalize_image

# Fixtures inspired by: https://gist.github.com/kissgyorgy/e2365f25a213de44b9a2
# Above patterns didn't quite work with how session __factory is shared.  Curious how I could improve my pattern as it
# feels a little more closed minded than this would be


FAKE_LABEL = 'fake_label'


@pytest.fixture(scope='session')
def db_init():
    # global_init builds tables and populates session factory
    global_init('', echo=True)


@pytest.fixture
def dummy_image(tmpdir):
    filename = os.path.join(tmpdir, 'raw_image.npy')
    np.save(filename, np.empty((500, 500)))
    return filename


@pytest.fixture
def db_with_image(dummy_image):
    m = Metadata()
    m.file_raw = dummy_image
    m.label = FAKE_LABEL
    s = create_session()
    s.add(m)
    s.commit()
    s.close()

    # Not sure why, but if I don't close and reopen the session I get a thread error.
    s = create_session()
    m_objs = s.query(Metadata).all()
    print([str(m) for m in m_objs])
    s.close()


def test_normalize_image_local(db_with_image, tmpdir, db_init):  # metadata_id, normalized_storage_location
    """
    Tests that normalize_image successfully loads the image from metadata_db, normalizes, and returns it to new store

    Output metadata and image are both inspected

    Future: Feels very scattered.  Break into a few tests?
    """
    # Verbose way to get m_id just in case we change the data creation routines.  This ensures there's a single record
    s = create_session()
    all_records = s.query(Metadata).all()
    s.close()
    assert len(all_records) == 1
    assert all_records[0].file_normalized is None

    m_id = all_records[0].id
    normalize_image(m_id, tmpdir)

    s = create_session()
    all_records = s.query(Metadata).all()
    s.close()
    assert len(all_records) == 1

    normalized_image_filename = all_records[0].file_normalized
    assert normalized_image_filename == os.path.join(tmpdir, f"{FAKE_LABEL}_{m_id}.npy")

    normalized_image = np.load(normalized_image_filename)
    assert normalized_image.shape == (28, 28)
