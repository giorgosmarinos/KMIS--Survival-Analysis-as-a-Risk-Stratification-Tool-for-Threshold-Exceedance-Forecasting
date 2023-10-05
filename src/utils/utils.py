import json
import pandas as pd 
import numpy as np 
import requests
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import norm, laplace, logistic, gumbel_r, lognorm, cauchy, genextreme
import statsmodels.api as sm


def mark_above_threshold(df, col_name, threshold, mark_col_name):
    """
    Given a Pandas DataFrame, a column name, a threshold value and a new column name,
    iterates through the values of the column and writes 1 to the new column if the
    value is above the threshold, and 0 otherwise.
    """
    df[mark_col_name] = df[col_name].apply(lambda x: 1 if x > threshold else 0)
    return df



def remove_invalid_observations(X: pd.DataFrame,
                                    y: pd.Series,
                                    lag_columns: list,
                                    decision_thr: float):
        """
        Remove observations where the target variable already exceeds the decision threshold.

        Parameters:
        - X (pd.DataFrame): Predictor variables as a DataFrame.
        - y (pd.Series or np.ndarray): Target variable.
        - lag_columns (list of str): Predictor columns relative to the target variable (lags).
        - decision_thr (float): Decision threshold.

        Returns:
        - X_t (pd.DataFrame): Processed predictor variables after removing invalid observations.
        - y_t (pd.Series or np.ndarray): Processed target variable after removing invalid observations.

        Example Usage:
        X = pd.DataFrame({'Predictor1': [1, 2, 3, 4, 5], 'Predictor2': [0.1, 0.2, 0.3, 0.4, 0.5]})
        y = np.array([0, 0, 1, 1, 1])
        lag_columns = ['Predictor1']
        decision_thr = 0.5
        X_processed, y_processed = remove_invalid_observations(X, y, lag_columns, decision_thr)
        """


        if isinstance(y, pd.Series):
            y = y.values

        idx_to_kp = ~(X[lag_columns] >= decision_thr).any(axis=1)

        X_t = X.loc[idx_to_kp, :].reset_index(drop=True).copy()
        y_t = y[idx_to_kp]

        return X_t, y_t



def remove_invalid_observations(X: pd.DataFrame,
                                    y: pd.Series,
                                    lag_columns: list,
                                    decision_thr: float):
        """
        Remove observations where the target variable already exceeds the decision threshold.

        Parameters:
        - X (pd.DataFrame): Predictor variables as a DataFrame.
        - y (pd.Series or np.ndarray): Target variable.
        - lag_columns (list of str): Predictor columns relative to the target variable (lags).
        - decision_thr (float): Decision threshold.

        Returns:
        - X_t (pd.DataFrame): Processed predictor variables after removing invalid observations.
        - y_t (pd.Series or np.ndarray): Processed target variable after removing invalid observations.

        Example Usage:
        X = pd.DataFrame({'Predictor1': [1, 2, 3, 4, 5], 'Predictor2': [0.1, 0.2, 0.3, 0.4, 0.5]})
        y = np.array([0, 0, 1, 1, 1])
        lag_columns = ['Predictor1']
        decision_thr = 0.5
        X_processed, y_processed = remove_invalid_observations(X, y, lag_columns, decision_thr)
        """


        if isinstance(y, pd.Series):
            y = y.values

        idx_to_kp = ~(X[lag_columns[-1]] >= decision_thr)#.any(axis=1)

        X_t = X.loc[idx_to_kp, :].copy()#.reset_index(drop=True).copy()
        y_t = y[idx_to_kp]

        return X_t, y_t



def identify_sharp_increases(df, col_name, time_window, threshold, increase_perc):
    """
    Identify sharp increases in the values of a specified column in a pandas DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the column of interest.
    - col_name (str): The name of the column to analyze.
    - time_window (int): The size of the time window for calculating the mean of past values.
    - threshold (float): The threshold below which values are considered irrelevant.
    - increase_perc (float): The percentage by which the recent value must exceed the mean to be considered a sharp increase.

    Returns:
    - sharp_increase_indices (list): A list of indices where sharp increases in the values of the column occur.

    Example Usage:
    import pandas as pd
    data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
             'Value': [5.0, 5.2, 6.0, 5.8, 5.5]}
    df = pd.DataFrame(data)
    col_name = 'Value'
    time_window = 3
    threshold = 5.0
    increase_perc = 0.1
    sharp_indices = identify_sharp_increases(df, col_name, time_window, threshold, increase_perc)
    """
    
    # Initialize an empty list to store the indices of sharp increases
    sharp_increase_indices = []
    
    # Get the values of the specified column as a numpy array
    col_values = df[col_name].values
    
    # Iterate through the values of the column, starting at index 6
    for i in range(time_window, len(col_values)):
        
        # Get the last 6 values of the column
        last_n_values = col_values[i-time_window:i]
        
        # Get the most recent value of the column
        recent_value = col_values[i]
        
        # Check if the current value is below 10
        if recent_value < threshold:
            
            # If so, exclude this index and continue to the next iteration
            continue
        
        # Calculate the mean of the last 6 values
        mean_last_n = last_n_values.mean()

        #alternative - weighted average
        #weights = [i+1 for i in range(len(last_n_values))]
        
        #give emphasis to the last one 
        #weights[-1] *= 5
        #print(weights)

        # Calculate the weighted average
        #mean_last_n = np.average(last_n_values, weights=weights)


        # Check if the most recent value is greater than the mean of the last 6 values
        if recent_value > mean_last_n:
            
            # Check if the most recent value is at least 50% greater than the mean of the last 6 values
            if recent_value >= mean_last_n * increase_perc:
                
                # If the conditions are met, add the index to the list of sharp increases
                sharp_increase_indices.append(i)
                print('the recent value is:',recent_value, 'while the mean value of the time window is',mean_last_n ,'and the increase is:',mean_last_n * increase_perc, 'ADDED')

            else:
                print('the recent value is:',recent_value, 'while the mean value of the time window is',mean_last_n ,'and  the increase is:',mean_last_n * increase_perc, 'NOT ADDED')
    
    return sharp_increase_indices


def insert_metrics_from_foodakai(predicted_historical_values, y_train, y_val, exceedance_prob, Column_name, THRESHOLD, timesteps, foodakai_model,
                       test_size, shuffle, invalid_observations_removal, 
                       no_of_columns):
    """
    Calculate and insert various metrics for different probability distributions into a final list.

    Parameters:
    - predicted_historical_values (array-like): Predicted historical values.
    - y_train (array-like): Training target values.
    - y_val (array-like): Validation target values.
    - exceedance_prob (array-like): Exceedance probability values.
    - Column_name (str): Name of the column or feature.
    - THRESHOLD (float): Threshold value for exceedance.
    - timesteps (int): Number of time steps.
    - foodakai_model (str): Foodakai model name or identifier.
    - test_size (float): Size of the test dataset.
    - shuffle (bool): Whether to shuffle data.
    - invalid_observations_removal (bool): Whether to remove invalid observations.
    - no_of_columns (int): Number of columns in the dataset.

    Returns:
    - final_list (list of lists): A list containing metric results for different probability distributions.

    Example Usage:
    predicted_historical_values = [0.2, 0.3, 0.4, 0.5]
    y_train = [0, 1, 0, 1]
    y_val = [0, 1, 1, 0]
    exceedance_prob = [0.25, 0.35, 0.45, 0.55]
    Column_name = 'Feature'
    THRESHOLD = 0.5
    timesteps = 10
    foodakai_model = 'Model-A'
    test_size = 0.2
    shuffle = True
    invalid_observations_removal = False
    no_of_columns = 5
    metrics_list = insert_metrics_from_foodakai(predicted_historical_values, y_train, y_val, exceedance_prob, Column_name, THRESHOLD, timesteps, foodakai_model, test_size, shuffle, invalid_observations_removal, no_of_columns)
    """
     
    distributions = [genextreme, norm, laplace, logistic, gumbel_r, lognorm, cauchy]

    final_list = []

    for distribution in distributions:


        std = y_train.std()
        if distribution == lognorm:
            exceedance_prob = np.asarray([1 - distribution.cdf(THRESHOLD, loc=x_, s=std) for x_ in predicted_historical_values])
        elif distribution == genextreme:
            exceedance_prob = np.asarray([1 - distribution.cdf(THRESHOLD, c = -0.1, loc=x_, scale=std) for x_ in predicted_historical_values])
        elif distribution != lognorm:
            exceedance_prob = np.asarray([1 - distribution.cdf(THRESHOLD, loc=x_, scale=std) for x_ in predicted_historical_values])


        y_val_ = (y_val > THRESHOLD).astype(int)
        y_pred_binary = np.where(exceedance_prob >= 0.5, 1, 0)

        # Calculate metrics
        accuracy = accuracy_score(y_val_, y_pred_binary)
        precision = precision_score(y_val_, y_pred_binary)
        recall = recall_score(y_val_, y_pred_binary)
        f1 = f1_score(y_val_, y_pred_binary)
        roc_auc = roc_auc_score(y_val_, exceedance_prob)

        final_list.append([Column_name, THRESHOLD, timesteps, distribution.name, foodakai_model, 
                            str(int(np.round(accuracy, 2)*100))+str('%'), 
                            str(int(np.round(precision, 2)*100))+str('%'), 
                            str(int(np.round(recall, 2)*100))+str('%'), 
                            str(int(np.round(f1, 2)*100))+str('%'), 
                            str(int(np.round(roc_auc, 2)*100))+str('%'), 
                           test_size, shuffle, invalid_observations_removal, 
                       no_of_columns])
        
    return final_list


def insert_metrics_from_foodakai_simple_threshold(predicted_historical_values, y_train, y_val, exceedance_prob, 
                       Column_name, THRESHOLD, timesteps, foodakai_model,
                       test_size, shuffle, invalid_observations_removal, 
                       no_of_columns):
    """
    Calculate and insert metrics for a simple threshold into a final list.

    Parameters:
    - predicted_historical_values (array-like): Predicted historical values.
    - y_train (array-like): Training target values.
    - y_val (array-like): Validation target values.
    - exceedance_prob (array-like): Exceedance probability values.
    - Column_name (str): Name of the column or feature.
    - THRESHOLD (float): Threshold value for exceedance.
    - timesteps (int): Number of time steps.
    - foodakai_model (str): Foodakai model name or identifier.
    - test_size (float): Size of the test dataset.
    - shuffle (bool): Whether to shuffle data.
    - invalid_observations_removal (bool): Whether to remove invalid observations.
    - no_of_columns (int): Number of columns in the dataset.

    Returns:
    - final_list (list): A list containing metric results for the simple threshold.

    Example Usage:
    predicted_historical_values = [0.2, 0.3, 0.4, 0.5]
    y_train = [0, 1, 0, 1]
    y_val = [0, 1, 1, 0]
    exceedance_prob = [0.25, 0.35, 0.45, 0.55]
    Column_name = 'Feature'
    THRESHOLD = 0.5
    timesteps = 10
    foodakai_model = 'Model-A'
    test_size = 0.2
    shuffle = True
    invalid_observations_removal = False
    no_of_columns = 5
    metrics_list = insert_metrics_from_foodakai_simple_threshold(predicted_historical_values, y_train, y_val, exceedance_prob, Column_name, THRESHOLD, timesteps, foodakai_model, test_size, shuffle, invalid_observations_removal, no_of_columns)
    """

    y_val_ = (y_val > THRESHOLD).astype(int)
    y_pred_binary = (predicted_historical_values > THRESHOLD).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_val_, y_pred_binary)
    precision = precision_score(y_val_, y_pred_binary)
    recall = recall_score(y_val_, y_pred_binary)
    f1 = f1_score(y_val_, y_pred_binary)
    roc_auc = roc_auc_score(y_val_, exceedance_prob)

    final_list = [Column_name, THRESHOLD, timesteps, 'None', foodakai_model, 
                        str(int(np.round(accuracy, 2)*100))+str('%'), 
                        str(int(np.round(precision, 2)*100))+str('%'), 
                        str(int(np.round(recall, 2)*100))+str('%'), 
                        str(int(np.round(f1, 2)*100))+str('%'), 
                        str(int(np.round(roc_auc, 2)*100))+str('%'), 
                        test_size, shuffle, invalid_observations_removal, 
                        no_of_columns]
        
    return final_list