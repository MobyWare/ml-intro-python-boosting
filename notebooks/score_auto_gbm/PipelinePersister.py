import cPickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from score_auto_gbm.FeatureTransformer import FeatureTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from datetime import datetime


class PipelinePersister(object):
    """
        Class that helps create and persist models using pickle
    """

    def __init__(self):
        pass

    def fit(self, features, target, model=None, pipleline=None, use_grid_search=True, grid_search_params=None, grid_verbosity=0):
        local_pipe = None
        local_model = None
        local_params = []
        predictor = None

        if features is None or target is None:
            raise ValueError

        if model is None:
            local_model = GradientBoostingRegressor(learning_rate=0.1, random_state=1234)
        else:
            local_model = model

        if pipleline is None:
            local_pipe = Pipeline([('preprocess', FeatureTransformer())])
        else:
            local_pipe = pipleline

        if model is not None:
            local_pipe.steps.extend([('custom', local_model)])
        else:
            local_pipe.steps.extend([('gbm', local_model)])

        if not use_grid_search:
            predictor = local_pipe.fit(features, target)
        else:
            if grid_search_params is None:
                local_params = dict(gbm__n_estimators=[50, 100, 150, 200], gbm__max_depth=[5, 6, 7, 8, 9, 10])
            else:
                local_params = grid_search_params

            predictor = GridSearchCV(local_pipe, local_params, cv=5, scoring=make_scorer(mean_squared_error), verbose=grid_verbosity).fit(features, target).best_estimator_

        # Return any preprocessing objects. Needed for scaling later.
        return predictor, local_pipe.named_steps['preprocess']

    def save(self, model, path=None):
        if model is None:
            raise ValueError

        if path is None:
            path = "model-" + datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(path, "wb") as pickle_file:
            cPickle.dump(model, pickle_file)

    def load(self, path):
        with open(path, "rb") as pickle_file:
            return cPickle.load(pickle_file)

