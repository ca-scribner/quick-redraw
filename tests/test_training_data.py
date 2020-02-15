import pytest

from quick_redraw.data.metadata_db_session import global_init, create_session
from quick_redraw.data.training_data import TrainingData


@pytest.fixture(scope='session')
def db_init():
    # global_init builds tables and populates session factory
    global_init('', echo=True)


def test_training_data_insert(db_init):
    s = create_session()

    train = ['1', '2', '3']
    test = ['a', 'b']
    class_names = ['cat', 'dog']

    td = TrainingData(train=train, test=test, class_names=class_names)
    s.add(td)
    s.commit()

    td_out = s.query(TrainingData).first()

    for attr in ['id', 'train', 'test', 'class_names']:
        assert getattr(td, attr) == getattr(td_out, attr)

    assert td == td_out
