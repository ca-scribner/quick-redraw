raise NotImplementedError("Need to refactor.  Uses old schema")

import os

from quick_redraw.data.image_record import ImageRecord
from quick_redraw.data.db_session import global_init, create_session
from quick_redraw.data.training_data import TrainingData
# from quick_redraw.data.training_data_record import TrainingDataRecord

db_file = './training_data_db_inserts.sqlite'

print("DELETING OLD TEMP DB")
os.remove(db_file)

global_init(db_file)

image = ImageRecord(label='cat', file_raw='raw.png', file_normalized='norm.png')

tdrs = [
    # TrainingDataRecord(),
    # TrainingDataRecord(),
    # TrainingDataRecord(),
]

tdrs[0].image = image

index_to_label = ['cat', 'dog']
label_to_index = {'cat': 0, 'dog': 1}

td = TrainingData(index_to_label=index_to_label, label_to_index=label_to_index)
td.train.extend(tdrs)

s = create_session()

s.add(td)

s.commit()