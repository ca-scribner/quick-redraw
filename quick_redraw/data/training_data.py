from typing import List

import sqlalchemy as sa
import sqlalchemy.orm
import datetime

from sqlalchemy import ForeignKey

from quick_redraw.data.modelbase import SqlAlchemyBase
from quick_redraw.data.big_integer_type import BigIntegerType
from quick_redraw.data.image_record import ImageRecord

# Association table lets us use one or more ImageRecord in any TrainingData.
# Note: with backref=parent below, deletions on one side of this relationship are propagated to the other.  So if we
# delete an ImageRecord then all TrainingData's that use it will have it removed.
# Great for live usage, but terrible for knowing what we invoked in the past.
training_data_image_record_association = sa.Table("training_data_image_record_association", SqlAlchemyBase.metadata,
                                                  sa.Column('training_data_id', BigIntegerType,
                                                            ForeignKey('training_data.id')),
                                                  sa.Column('image_record_id', BigIntegerType,
                                                            ForeignKey('image_record.id')),
                                                  )

testing_data_image_record_association = sa.Table("testing_data_image_record_association", SqlAlchemyBase.metadata,
                                                 sa.Column('training_data_id_', BigIntegerType,
                                                           ForeignKey('training_data.id')),
                                                 sa.Column('image_record_id_', BigIntegerType,
                                                           ForeignKey('image_record.id')),
                                                 )


class TrainingData(SqlAlchemyBase):
    __tablename__ = "training_data"

    id: int = sa.Column(BigIntegerType, primary_key=True, autoincrement=True)
    index_to_label: list = sa.Column(sa.JSON)
    label_to_index: dict = sa.Column(sa.JSON)
    created_date: datetime.datetime = sa.Column(sa.DateTime, default=datetime.datetime.now, index=True)

    # See note on backref above at association table
    # Backref puts an attribute on the ImageRecord python object, so we need unique names.  We can see these attributes
    # when instancing an ImageRecord then looking at list(img.__mapper__.attrs)
    # First arg of relationship can be the name of a data object or the actual class
    training_images = sa.orm.relationship(ImageRecord, secondary=training_data_image_record_association,
                                          backref='parents_training_images')
    # (this first arg is equivalent to above)
    testing_images = sa.orm.relationship("ImageRecord", secondary=testing_data_image_record_association,
                                         backref='parents_testing_images')

    def __repr__(self):
        return f"TrainingData id={self.id}, " \
               f"training_images={self.training_images}, " \
               f"testing_images={self.testing_images}, " \
               f"index_to_label={self.index_to_label}, " \
               f"label_to_index={self.label_to_index}, " \
               f"created_date={self.created_date}"
