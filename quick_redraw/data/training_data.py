from typing import List

import sqlalchemy as sa
import datetime

from quick_redraw.data.modelbase import SqlAlchemyBase
from quick_redraw.data.big_integer_type import BigIntegerType


class TrainingData(SqlAlchemyBase):
    __tablename__ = "training_data"

    id: int = sa.Column(BigIntegerType, primary_key=True, autoincrement=True)

    # FUTURE: Train/Test image lists stored as json lists.  Should use a separate table for a list of foreign keys
    # eg: https://stackoverflow.com/questions/3070384/how-to-store-a-list-in-a-column-of-a-database-table
    # Should I use sa.JSON or just text and interpret as json myself?  Not everything supports JSON natively
    train: list = sa.Column(sa.JSON)
    test: list = sa.Column(sa.JSON)
    class_names: list = sa.Column(sa.JSON)
    created_date: datetime.datetime = sa.Column(sa.DateTime, default=datetime.datetime.now, index=True)

    def __str__(self):
        return f"TrainingData id={self.id}, train={self.train}, test={self.test}, class_map={self.class_names}"
