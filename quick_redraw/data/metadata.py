import sqlalchemy
from sqlalchemy import BigInteger
from sqlalchemy.dialects import sqlite
import datetime

from quick_redraw.data.modelbase import SqlAlchemyBase

# sqlite does not allow BigInteger as a primary key with autoincrement.  Use an integer for sqlite (local testing)
# but BigInteger elsewhere
BigIntegerType = BigInteger()
BigIntegerType = BigIntegerType.with_variant(sqlite.INTEGER(), 'sqlite')


class Metadata(SqlAlchemyBase):
    __tablename__ = "metadata"

    id: int = sqlalchemy.Column(BigIntegerType, primary_key=True, autoincrement=True)
    # id: int = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    created_date: datetime.datetime = sqlalchemy.Column(sqlalchemy.DateTime, default=datetime.datetime.now, index=True)
    label: str = sqlalchemy.Column(sqlalchemy.String)  # Could be an enumerator?  Or categorical?  Supported?
    file_raw: str = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    file_normalized: str = sqlalchemy.Column(sqlalchemy.String, nullable=True)

    def __str__(self):
        return f"Metadata id={self.id}, label={self.label}, created_date={self.created_date}, file_raw={self.file_raw}," \
               f" file_normalized={self.file_normalized}"
