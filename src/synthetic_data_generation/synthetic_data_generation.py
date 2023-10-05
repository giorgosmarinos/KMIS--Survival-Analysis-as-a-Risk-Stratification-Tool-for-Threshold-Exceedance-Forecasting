import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def generate_simulated_multivariate_time_series(n_steps=1000, n_series=3):
    """
    Generates simulated multivariate time series data with trend, seasonality, and random extreme values.
    
    Parameters:
        n_steps (int): The number of time steps in the generated data.
        n_series (int): The number of series in the multivariate data.
    
    Returns:
        pd.DataFrame: A Pandas DataFrame containing the simulated time series data.
    """
    # Set random seed for reproducibility
    np.random.seed(123)

    # Create a random covariance matrix for the series
    cov_matrix = np.random.rand(n_series, n_series)

    # Generate the random series data
    data = np.random.multivariate_normal(mean=[0]*n_series, cov=cov_matrix, size=n_steps)

    # Add some random extreme values to each series
    for i in range(n_series):
        rand_indices = np.random.choice(n_steps, size=20, replace=False)
        data[rand_indices, i] += np.random.normal(loc=10, scale=2, size=20)

    # Add trend and seasonality to each series
    freq = 'M'
    dates = pd.date_range(start='2022-01-01', periods=n_steps, freq=freq)
    for i in range(n_series):
        # Generate a random trend component that changes over time
        trend = np.zeros(n_steps)
        for j in range(1, n_steps):
            trend[j] = trend[j-1] + np.random.normal(scale=0.5)

        # Add a random offset to the trend
        offset = np.random.normal(scale=1)
        trend += (i+1)*np.linspace(0, 10, n_steps) + offset + np.random.normal(scale=0.5, size=n_steps)

        seasonality = (i+1)*np.sin(np.linspace(0, 6*np.pi, n_steps)) + np.random.normal(scale=0.2, size=n_steps)
        data[:, i] += trend + seasonality

    # Decompose the data into trend, seasonality, and residual components
    df = pd.DataFrame(data=data, index=dates, columns=[f'Series {i+1}' for i in range(n_series)])
    decomposition = seasonal_decompose(df.values, model='additive', period=12)

    # Get the trend, seasonality, and residual components
    trend = decomposition.trend
    seasonality = decomposition.seasonal
    residual = decomposition.resid

    # Add the trend, seasonality, and residual components to the data
    data = trend + seasonality + residual

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data=data, index=dates, columns=[f'Series {i+1}' for i in range(n_series)])

    return df

def plot_multivariate_time_series(df):
    """
    Plots the simulated multivariate time series data.
    
    Parameters:
        df (pd.DataFrame): The Pandas DataFrame containing the simulated time series data.
    """
    # Plot the data
    df.plot(figsize=(12, 6), title='Simulated Multivariate Time Series Data')
    plt.xlabel('Time')
    plt.show()

# Example usage:
# simulated_data = generate_simulated_multivariate_time_series()
# plot_multivariate_time_series(simulated_data)
