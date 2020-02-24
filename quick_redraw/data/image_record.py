import sqlalchemy as sa
import datetime

from quick_redraw.data.modelbase import SqlAlchemyBase
from quick_redraw.data.big_integer_type import BigIntegerType


class ImageRecord(SqlAlchemyBase):
    __tablename__ = "image_record"

    id: int = sa.Column(BigIntegerType, primary_key=True, autoincrement=True)
    # id: int = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    created_date: datetime.datetime = sa.Column(sa.DateTime, default=datetime.datetime.now, index=True)
    label: str = sa.Column(sa.String)  # Could be an enumerator?  Or categorical?  Supported?
    file_raw: str = sa.Column(sa.String, nullable=True)
    file_normalized: str = sa.Column(sa.String, nullable=True)

    def __repr__(self):
        return f"Metadata id={self.id}; label={self.label}; created_date={self.created_date}; " \
               f"file_raw={self.file_raw}; file_normalized={self.file_normalized}"
