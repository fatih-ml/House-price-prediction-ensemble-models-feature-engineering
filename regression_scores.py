from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np

def calculate_score(y_train, y_train_pred, y_test, y_pred, model_name, cv_scores=None):
    
    """
    this function is serving for two purposes
    1. helping me the main function
    2. individual score calculation and evaluation of models
    model_name(string)
    when calling individually, dont specify the cv_scores
    """
    
    model_name_short = model_name[:5]
    
    scores = {
        f'{model_name_short}_train': {
            "R2" : r2_score(y_train, y_train_pred),
            "-mae" : mean_absolute_error(y_train, y_train_pred),
            "-mse" : mean_squared_error(y_train, y_train_pred),
            "-rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        f'{model_name_short}_test':  {
            "R2" : r2_score(y_test, y_pred),
            "-mae" : mean_absolute_error(y_test, y_pred),
            "-mse" : mean_squared_error(y_test, y_pred),
            "-rmse" : np.sqrt(mean_squared_error(y_test, y_pred))
        },
    }
    
    # Try-except block for the cross-validation scores
    try:
        cv_scores_mean = cv_scores.mean()
        scores[f'{model_name_short}_CV'] = {
            "R2": cv_scores_mean[0],
            "-mae": cv_scores_mean[1],
            "-mse": cv_scores_mean[2],
            "-rmse": cv_scores_mean[3]
        }
    except Exception as e:
        print(f"Error calculating CV scores: {e}")
    
    return pd.DataFrame(scores)



def calculate_scores(pipelines, X_train, X_test, y_train, y_test):
    """
    This function is taking a dictionary of pipelines
    and calculate 4 metrics related to regression models on both train and test dataset
    including cross validation scores
    returns a dataframe of comparison
    
    pipelines(dict): dictionary of pipelines
    returns pd.DataFrame
    """

    
    # make an empty dictionary and a dataFrame to stack the results   
    model_scores_dict = {}
    counter_concat = 0
    model_scores = pd.DataFrame()
    scoring_metrics= ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error' ]
    
    # iterate through our stored pipelines in pipeline dictionary
    for model_name, pipeline in pipelines.items():
        
        # fit the pipelines
        pipeline.fit(X_train, y_train)

        # make predictions
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)

            
        cv_scores = pd.DataFrame(cross_validate(pipeline, X_train, y_train, scoring=scoring_metrics)).iloc[:,2:]
        
        # calculate the scores
        scores_pipeline = calculate_score(y_train, y_train_pred, y_test, y_pred, model_name, cv_scores)
 
        # combine in a dataframe for comparison
        model_scores = (pd.concat([model_scores, scores_pipeline], axis=1))
    
        
    return model_scores


