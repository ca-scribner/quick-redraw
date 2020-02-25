from itertools import cycle

import pytest

from quick_redraw.data.db_session import global_init, create_session, global_forget
from quick_redraw.data.image_record import ImageRecord
from quick_redraw.data.training_data import TrainingData
from quick_redraw.etl.train_test_split import create_training_data_from_image_db
from quick_redraw.services.metadata_service import find_records_with_label_normalized


@pytest.fixture
def db_init():
    # global_init builds tables and populates session factory
    global_init('', echo=True)
    yield
    global_forget()


@pytest.fixture
def db_with_images(db_init):
    s = create_session()
    n_images = 10
    c = cycle(['cat', 'dog'])
    imgs = [ImageRecord(label=next(c), file_raw='raw.png') for i in range(n_images)]
    for i in range(0, len(imgs) // 2):
        imgs[i].file_normalized = 'norm.png'
    s.add_all(imgs)
    s.commit()


def test_training_data_insert(db_with_images):
    s = create_session()

    records = find_records_with_label_normalized()

    index_to_label = ['cat', 'dog']
    label_to_index = {0: 'cat', 1: 'dog'}

    td = TrainingData(index_to_label=index_to_label, label_to_index=label_to_index)
    td.training_images.extend(records[:2])
    td.testing_images.extend(records[2:])

    s.add(td)
    s.commit()

    td_out = s.query(TrainingData).first()

    assert len(td_out.training_images) == 2

    for attr in ['id', 'training_images', 'testing_images', 'index_to_label', 'label_to_index']:
        assert getattr(td, attr) == getattr(td_out, attr)

    assert td == td_out


def test_train_test_split(db_with_images):
    create_training_data_from_image_db(test_size=0.4, random_state=42)

    s = create_session()
    tds = s.query(TrainingData).all()

    assert len(tds) == 1

    td = tds[0]
    assert len(td.training_images) == 3
    assert len(td.testing_images) == 2
