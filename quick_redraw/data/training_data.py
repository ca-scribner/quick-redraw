from typing import List

import sqlalchemy as sa
import datetime

from quick_redraw.data.modelbase import SqlAlchemyBase
from quick_redraw.data.big_integer_type import BigIntegerType


class TrainingData(SqlAlchemyBase):
    __tablename__ = "training_data"

    id: int = sa.Column(BigIntegerType, primary_key=True, autoincrement=True)

    # FUTURE: Train and Test records should be stored here as references to another table that keeps each individual
    # record.  That table would reference original_metadata_id, embedded_label.  Then here we'd also have the
    # labeel_to_index and index_to_label.
    # Might need this too?
    # eg: https://stackoverflow.com/questions/3070384/how-to-store-a-list-in-a-column-of-a-database-table
    train: list = sa.Column(sa.JSON)
    test: list = sa.Column(sa.JSON)
    labels_as_index: list = sa.Column(sa.JSON)
    index_to_label: list = sa.Column(sa.JSON)
    label_to_index: dict = sa.Column(sa.JSON)
    created_date: datetime.datetime = sa.Column(sa.DateTime, default=datetime.datetime.now, index=True)

    # def __str__(self):
    #     return f"TrainingData id={self.id}, train={self.train}, test={self.test}, class_map={self.index_to_class_name}"
