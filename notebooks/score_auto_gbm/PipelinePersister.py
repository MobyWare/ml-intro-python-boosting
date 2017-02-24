import cPickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from score_auto_gbm.FeatureTransformer import FeatureTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from datetime import datetime

# Class that helps create and persist models using pickle
class PipelinePersister(object):
    def __init__(self):
        pass

    def fit(self, features, target, model=None, pipleline=None, useGridSearch=True, gridSearchParams=None, gridVerbosity=0):
        localPipe = None
        localModel = None
        localParams = []
        predictor = None

        if features is None or target is None:
            raise ValueError

        if model is None:
            localModel = GradientBoostingRegressor(learning_rate=0.1, random_state=1234) 
        else:
            localModel = model

        if pipleline is None:
            localPipe = Pipeline([('preprocess', FeatureTransformer())])
        else: 
            localPipe = pipleline

        if model is not None:
            localPipe.steps.extend([('custom', localModel)])
        else:
            localPipe.steps.extend([('gbm', localModel)])
        
        if not useGridSearch:
            predictor = localPipe.fit(features, target)
        else:
            if gridSearchParams is None:
                localParams = dict(gbm__n_estimators = [50, 100, 150, 200], gbm__max_depth = [5, 6, 7, 8, 9, 10])
            else:
                localParams = gridSearchParams
            
            predictor = GridSearchCV(localPipe, localParams, cv = 5, scoring = make_scorer(mean_squared_error), verbose=gridVerbosity).fit(features, target).best_estimator_
        
        return predictor

    def save(self, model, path=None):
        if model is None:
            raise ValueError
        
        if path is None:
            path = "model-" + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(path, "wb") as pickle_file:
            cPickle.dump(model, pickle_file)

    def load(path):
        with open(path, "rb") as pickle_file:
            return cPickle.load(pickle_file)                             

