import cPickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from score_auto_gbm.FeatureTransformer import FeatureTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from datetime import datetime

# Class that return statistics about models
class ModelEvaluator(object):
    def __init__(self):
        pass
    
    def is_number(self, input_to_test):
        try:
            float(input_to_test)
            return True
        except ValueError:
            return False
        

    def get_predict_vs_actuals(self, model, features, actuals=None, preprocessor=None):
        result = []
        prediction_list = []
        
        if actuals is None:
            for row in features.to_dict(orient='records'):
                try:
                    prediction = list(model.predict(pd.DataFrame([row])))[0]
                except ValueError:
                    continue
                
                if self.is_number(prediction) == True:
                    prediction_list.extend([float(prediction)])

            result = zip(np.array(prediction_list), np.array([sum(prediction_list)/len(prediction_list)] * len(prediction_list)))
        else:            
            for idx, row in enumerate(features.to_dict(orient='records')):
                try:
                    prediction = list(model.predict(pd.DataFrame([row])))[0]
                except ValueError:
                    continue
                
                if self.is_number(actuals[idx]) == True and self.is_number(prediction) == True:
                    result.extend([(float(prediction), float(actuals[idx]))])

            if(len(result) < len(actuals)):
                print(
                    'Some predicions may have failed. # of successful predictions: %d, # of actual values: %d' % 
                    (len(actuals), len(result)))    

        return result   
    
    def get_rmse_summary(self, predction_tuple_array):
        total = len(predction_tuple_array)
        squared_error = map(lambda errorTuple: (errorTuple[0] - errorTuple[1])**2, predction_tuple_array)
        rmse = (sum(squared_error)/total)**0.5

        average_prediction = sum(map(lambda errorTuple: errorTuple[1], predction_tuple_array))/total
        standard_squared_error = map(lambda errorTuple: (average_prediction - errorTuple[1])**2, predction_tuple_array)
        standard_rmse = (sum(standard_squared_error)/total)**0.5

        return rmse, standard_rmse, total

    def get_feature_importance_tuples(self, model, features, sort=True):
        if(model.feature_importances_ is None):
            raise ValueError('The model must have important features method.')
        else:
            if(sort):
                return zip(features.columns, model.feature_importances_).sort(key=lambda x:-x[1])
            else:
                return zip(features.columns, model.feature_importances_)