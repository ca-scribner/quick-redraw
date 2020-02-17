from collections import namedtuple

import numpy as np
from ray.tune.schedulers import MedianStoppingRule
from sklearn.pipeline import Pipeline
from tune_sklearn.tune_search import TuneGridSearchCV

from quick_redraw.data.metadata_db_session import global_init
from quick_redraw.services.image_storage_service import load_drawings
from sklearn.svm import SVC

from scipy.stats import loguniform


# Future: What is a more flexible way of defining these models?  Need something that lets me define them in a config
# file that can be imported here cleanly
def svm_pipeline():
    pl = Pipeline([
        ('svm', SVC()),
    ])
    return pl


def svm_parameter_space():
    param_grid = [
        {
            'kernel': ['linear'],
            'C': loguniform(0.001, 10),
        },
        {
            'kernel': ['rbf'],
            'C': loguniform(0.1, 1000),
            'gamma': loguniform(0.0001, 1.0),
        }
    ]
    return param_grid


def train():
    # NOTE: Can use ray through joblib, but the feature is not included in the pip install'd code (it is ~2 weeks old)
    # see here: https://ray.readthedocs.io/en/latest/joblib.html
    # and pip install the nightly wheel from here: https://ray.readthedocs.io/en/latest/installation.html

    n_iter = 10
    cv = 5

    label_drawing_tuples = load_drawings(storage_location='normalized')
    if not label_drawing_tuples:
        raise ValueError("No training drawings found in db")
    labels, drawings = zip(*label_drawing_tuples)

    drawings = np.asarray(drawings)
    labels = np.asarray(labels)

    print(f"drawings.shape = {drawings.shape}")
    print(f"labels.shape = {labels.shape}")

    pipeline_nt = namedtuple('pipeline', ['pipeline', 'parameter_space'])
    pipelines = {
        'svm': pipeline_nt(svm_pipeline(), svm_parameter_space()),
    }

    for name, pipeline_tuple in pipelines.items():
    #     rs = RandomizedSearchCV(case.pipeline, case.parameter_space, n_iter=n_iter, n_jobs=-1, refit=False, cv=cv)
        print(f"running {name}")
        temp_params = {
            'svm__gamma': [0.0001, 0.001],
            'svm__C': [1, 10],
        }
        tune_search = TuneGridSearchCV(pipeline_tuple.pipeline,
                                       temp_params,
                                       scheduler=MedianStoppingRule()
                                       )
        stuff = tune_search.fit(drawings, labels)

        pred = tune_search.predict(drawings)

        correct = 0
        for i in range(len(pred)):
            if pred[i] == labels[i]:
                correct += 1
        print(correct / len(pred))
        print(tune_search.cv_results_)




if __name__ == '__main__':
    global_init("./metadata.sqlite")
    train()
