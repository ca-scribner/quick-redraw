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


# Fixtures built from here: https://gist.github.com/kissgyorgy/e2365f25a213de44b9a2
from quick_redraw.etl.normalize import normalize_image


FAKE_LABEL = 'fake_label'

@pytest.fixture(scope='session')
def db_engine():
    return create_engine('sqlite://', echo=True)


@pytest.fixture(scope='session')
def db_tables(db_engine):
    """
    Builds all tables for this DB, then tears down after
    """
    SqlAlchemyBase.metadata.create_all(db_engine)
    yield
    SqlAlchemyBase.metadata.drop_all(db_engine)


@pytest.fixture
def db_session(db_engine, db_tables):
    # Not sure if we need to do this way or if we can do the same session-building methods from metadata_db_session.py
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    # Teardown
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def dummy_image(tmpdir):
    filename = os.path.join(tmpdir, 'raw_image.npy')
    np.save(filename, np.empty((500, 500)))
    return filename


@pytest.fixture
def db_with_image(db_session, dummy_image):
    m = Metadata()
    m.file_raw = dummy_image
    m.label = FAKE_LABEL
    db_session.add(m)
    db_session.commit()
    m_objs = db_session.query(Metadata).all()
    print([str(m) for m in m_objs])

    return db_session


def test_normalize_image(db_with_image, tmpdir):  # metadata_id, normalized_storage_location
    # global_init('')
    # s = create_session()
    # s.add(Metadata())
    # s.commit()
    # m_objs = s.query(Metadata).all()
    # print(m_objs)
    #
    # assert False

    # Need to mock np.save because we don't want to actually save images

    normalize_image(0, tmpdir)

    normalized_image_filename = db_with_image.query(Metadata).all()[0].file_normalized

    assert normalized_image_filename == os.join(tmpdir, f"{'fake_label'}_{0}.npy")

    normalized_image = np.load(normalized_image_filename)

    assert normalized_image.shape == (28, 28)
