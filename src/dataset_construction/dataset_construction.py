import json
import pandas as pd 
import numpy as np 
import requests
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import norm, laplace, logistic, gumbel_r, lognorm, cauchy, genextreme
import statsmodels.api as sm

def create_dataframe(num_lags, numbers, real_value, extreme_event, start, stop, duration):
    """
    Create a pandas DataFrame with lag columns and additional metadata.

    Parameters:
    - num_lags (int): Number of lag columns to create.
    - numbers (list): List of numerical values from which lag columns are derived.
    - real_value (list): List of real values to be included in the DataFrame.
    - extreme_event (list): List of extreme event indicators to be included.
    - start (list): List of start dates to be included.
    - stop (list): List of stop dates to be included.
    - duration (list): List of durations to be included.

    Returns:
    - df_new (pandas.DataFrame): A DataFrame containing lag columns, real_value,
      extreme_event, start_date, stop_date, and duration.

    Example Usage:
    num_lags = 3
    numbers = [10, 20, 30, 40, 50]
    real_value = [0.5, 0.6, 0.7, 0.8, 0.9]
    extreme_event = [False, False, True, True, False]
    start = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    stop = ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']
    duration = [2, 2, 2, 2, 2]
    df = create_dataframe(num_lags, numbers, real_value, extreme_event, start, stop, duration)
    """
    # Initialize a dictionary to store the lag columns
    lag_columns = {}

    # Create lag columns in the dictionary
    for lag in range(1, num_lags + 1):
        lag_columns[f'lag_{lag}'] = [arr[lag - 1] for arr in numbers]

    # Add other columns to the dictionary
    lag_columns['real_value'] = real_value
    lag_columns['extreme_event'] = extreme_event
    lag_columns['start_date'] = start
    lag_columns['stop_date'] = stop
    lag_columns['duration'] = duration

    # Convert the dictionary to a dataframe
    df_new = pd.DataFrame(lag_columns)

    return df_new


# the old name of this function was -> country_dataframe_cr
def duration_calculation(window_size, df, columns):
    """
    Calculate event durations and create a DataFrame with lag columns and additional metadata.

    Parameters:
    - window_size (int): Size of the time window for calculations.
    - df (pandas.DataFrame): DataFrame containing time series data.
    - columns (list): List of column names, where columns[0] corresponds to the time series data,
      and columns[1] corresponds to the event indicator.

    Returns:
    - df_new (pandas.DataFrame): A DataFrame containing lag columns, real_value,
      extreme_event, start_date, stop_date, and duration.

    Example Usage:
    window_size = 5
    data = pd.DataFrame({'Timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
                          'Value': [10, 20, 30, 40, 50],
                          'Event': [0, 0, 1, 1, 0]})
    columns = ['Value', 'Event']
    df = duration_calculation(window_size, data, columns)
    """
    #ts = df[df['country']==country]

    ts = df[columns].squeeze()
    print(ts)

    # Define the window size and the number of output steps
    output_steps = 1

    # Create input-output pairs from the time series using a sliding window
    numbers = []
    real_value = []
    extreme_event = []
    duration = []
    start = []
    stop = []

    start_date = ts.index[0]

    for i in range(len(ts)-window_size):
        
        numbers.append(ts[columns[0]][i:i+window_size].values)
        real_value.append(ts.iloc[i+window_size+output_steps-1][columns[0]])
        extreme_event.append(ts.iloc[i+window_size+output_steps-1][columns[1]])

        stop_date = ts.index[i+window_size+output_steps-1]

        if ts.iloc[i+window_size+output_steps-1][columns[1]] == 1:
        
          
          # Calculate the duration of the event
          duration.append((stop_date - start_date).days)

          #change the start date 
          start_date = ts.index[i+window_size+output_steps-1]

        else: 
          # Calculate the duration of the event
          duration.append((stop_date - start_date).days)

        # Extract the start and stop date of the time window
        start.append(start_date)
        stop.append(stop_date)
        

    # Convert the lists to a dataframe    
    df_new = create_dataframe(num_lags=window_size, 
                              numbers=numbers, 
                              real_value=real_value, 
                              extreme_event=extreme_event, 
                              start=start, 
                              stop=stop, 
                              duration=duration)

    return df_new


# the old name of this function was -> country_dataframe_cr
def alter_duration_calculation(window_size, df, columns):
    """
    Calculate and alternative duration values based on a sliding window.

    Parameters:
    - window_size (int): Size of the sliding window.
    - df (pandas.DataFrame): DataFrame containing time series data.
    - columns (list of str): List of column names to consider for calculations.

    Returns:
    - df_new (pandas.DataFrame): DataFrame with altered duration values.

    Example Usage:
    window_size = 5
    columns = ['ColumnA', 'ColumnB']
    df_new = alter_duration_calculation(window_size, df, columns)
    """
    #ts = df[df['country']==country]

    ts = df[columns].squeeze()
    print(ts)

    # Define the window size and the number of output steps
    output_steps = 1

    # Create input-output pairs from the time series using a sliding window
    numbers = []  #it is not needed, but still remains here in order to facilitate future changes
    real_value = [] #it is not needed, but still remains here in order to facilitate future changes
    extreme_event = []
    duration = []
    start = []
    stop = []

    start_date = ts.index[0]

    for i in range(len(ts)-window_size):
        
        numbers.append(ts[columns[0]][i:i+window_size].values) #it is not needed, but still remains here in order to facilitate future changes
        real_value.append(ts.iloc[i+window_size+output_steps-1][columns[0]]) #it is not needed, but still remains here in order to facilitate future changes
        extreme_event.append(ts.iloc[i+window_size+output_steps-1][columns[1]])

        stop_date = ts.index[i+window_size+output_steps-1]
          
        # Calculate the duration of the event
        duration.append((stop_date - start_date).days)

        # Extract the start and stop date of the time window
        start.append(start_date)
        stop.append(stop_date)
        

    # Convert the lists to a dataframe    
    df_new = create_dataframe(num_lags=window_size, 
                              numbers=numbers, 
                              real_value=real_value, 
                              extreme_event=extreme_event, 
                              start=start, 
                              stop=stop, 
                              duration=duration)

    return df_new