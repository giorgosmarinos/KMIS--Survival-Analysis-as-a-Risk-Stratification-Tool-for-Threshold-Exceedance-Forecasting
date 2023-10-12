import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from src.synthetic_data_generation.synthetic_data_generation import *
from src.utils.utils import *
from src.data_fetching.data_fetching import *
from src.dataset_construction.dataset_construction import *
import yaml

class DataPreprocessor:
    
    """
    A class for preprocessing time series data for machine learning and survival analysis tasks.

    Parameters:
    - data (DataFrame): The input data containing timestamped records and target columns.
    - timesteps (int): The number of consecutive time steps to include in each sequence.
    - target_column (str): The name of the target column to predict.
    - interval (str): The time interval for data aggregation, either "WEEK" or "MONTH".

    Methods:
    - create_supervised_data(data, timesteps, target_column, interval):
      Preprocesses the input data, aggregates it based on the specified interval, and creates supervised learning sequences.

    - train_test_split_(data, test_size_):
      Splits the preprocessed data into training and testing sets for machine learning tasks.
      
    - invalid_observations_removal_(X_train, X_test, y_train, y_test, THRESHOLD, invalid_observations_removal):
      Optionally removes invalid observations based on a threshold.

    Usage Example:
    data_processor = DataPreprocessing(my_data, timesteps=5, target_column='Incidents', interval='WEEK')
    X, y = data_processor.create_supervised_data()
    X_train, X_test, y_train, y_test, _, _, _, _, _, _, _, _ = data_processor.train_test_split_(test_size=0.2)
    X_train, X_test, y_train, y_test = data_processor.invalid_observations_removal_(X_train, X_test, y_train, y_test, THRESHOLD=0.5, invalid_observations_removal=True)
    """

    
    def __init__(self, data, apikey, headers, search_endpoint,
                 data_dir, data_ingredient, predictions_ingredient, data_hazard, predictions_hazard,
                 THRESHOLD_PERCENTILE, timesteps, random_state, Column_name, test_size_, shuffle_,
                 invalid_observations_removal_, interval, starting_date, end_date):
        
        self.data = data
        self.apikey = apikey
        self.headers = headers
        self.search_endpoint = search_endpoint
        self.data_dir = data_dir 
        self.data_ingredient = data_ingredient 
        self.predictions_ingredient = predictions_ingredient 
        self.data_hazard = data_hazard
        self.predictions_hazard = predictions_hazard
        self.THRESHOLD_PERCENTILE = THRESHOLD_PERCENTILE 
        self.timesteps = timesteps
        self.random_state = random_state
        self.Column_name = Column_name
        self.test_size_ = test_size_ 
        self.shuffle_ = shuffle_
        self.invalid_observations_removal_ = invalid_observations_removal_
        self.interval = interval
        self.starting_date = starting_date
        self.end_date = end_date
        
        self.validation_rows = len(data_prediction_request(self.apikey, 
                                                           self.headers, 
                                                           self.search_endpoint, 
                                                           self.data_dir, 
                                                           self.predictions_ingredient, 
                                                           self.predictions_hazard, 
                                                           self.interval, 
                                                           self.starting_date, 
                                                           self.end_date))


    def create_supervised_data(self):
        data = self.data.fillna(0)

        if self.interval == "WEEK":
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data['DayOffset'] = data.groupby('Date').cumcount() * 6
                data['Date_week'] = data['Date'] + pd.to_timedelta(data['DayOffset'], unit='D')
                data = data.drop(columns=['Date'])
                data.rename(columns={'Date_week': 'Date'}, inplace=True)
                data = data.set_index('Date')

        elif self.interval == "MONTH":
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')

        sequences = []
        labels = []

        for i in range(len(data) - self.timesteps):
            sequence = data[self.Column_name][i:i+self.timesteps]
            target = data[self.Column_name][i+self.timesteps]
            sequences.append(sequence.values)
            labels.append(target)

        self.X = pd.DataFrame(sequences, columns=[f'lag_{i+1}' for i in range(self.timesteps)])
        self.y = pd.Series(labels, name='y')

        return self.X, self.y

    def train_test_split(self, test_size_):
        data = self.data.copy()  # Make a copy to avoid modifying the original data
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

        if test_size_ == 0:
            X_train = self.X[:-self.validation_rows]
            y_train, y_test = self.y[:-self.validation_rows], self.y[-self.validation_rows:]
            X_test = self.X[-self.validation_rows:]

            THRESHOLD = np.quantile(y_train, self.THRESHOLD_PERCENTILE)
            data['extreme_event'] = data['Incidents'].apply(lambda x: 1 if x > THRESHOLD else 0)

            new_df = duration_calculation(self.timesteps, data, columns=[self.Column_name, 'extreme_event'])
            X_train_survival = new_df[:-self.validation_rows].drop(columns=['real_value', 'extreme_event', 'duration', 'start_date', 'stop_date'])
            X_test_survival = new_df[-self.validation_rows:].drop(columns=['real_value', 'extreme_event', 'duration', 'start_date', 'stop_date'])
            y_train_survival = Surv.from_dataframe(event='extreme_event', time='duration', data=new_df[:-self.validation_rows].drop(columns=['real_value']))
            y_test_survival = Surv.from_dataframe(event='extreme_event', time='duration', data=new_df[-self.validation_rows:].drop(columns=['real_value']))

            features_df = extract_features(data, self.timesteps)
            new_df_ext_feat_ = alter_duration_calculation(self.timesteps, data, columns=[self.Column_name, 'extreme_event'])
            new_df_ext_feat = new_df_ext_feat_.iloc[:, self.timesteps:]
            for i, column in enumerate(features_df.columns):
                new_df_ext_feat.insert(i, column, value=features_df[column])
            new_df_ext_feat.fillna(0, inplace=True)
            X_train_survival_ext_feat = new_df_ext_feat[:-self.validation_rows].drop(columns=['real_value', 'extreme_event', 'duration', 'start_date', 'stop_date'])
            X_test_survival_ext_feat = new_df_ext_feat[-self.validation_rows:].drop(columns=['real_value', 'extreme_event', 'duration', 'start_date', 'stop_date'])
            y_train_survival_ext_feat = Surv.from_dataframe(event='extreme_event', time='duration', data=new_df_ext_feat[:-self.validation_rows].drop(columns=['real_value']))
            y_test_survival_ext_feat = Surv.from_dataframe(event='extreme_event', time='duration', data=new_df_ext_feat[-self.validation_rows:].drop(columns=['real_value']))

            return X_train, X_test, y_train, y_test, THRESHOLD, X_train_survival, X_test_survival, y_train_survival, y_test_survival, X_train_survival_ext_feat, X_test_survival_ext_feat, y_train_survival_ext_feat, y_test_survival_ext_feat

        elif test_size_ != 0:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size_, shuffle=self.shuffle_, random_state=self.random_state)

            THRESHOLD = np.quantile(y_train, self.THRESHOLD_PERCENTILE)
            data['extreme_event'] = data['Incidents'].apply(lambda x: 1 if x > THRESHOLD else 0)

            new_df = duration_calculation(self.timesteps, data, columns=[self.Column_name, 'extreme_event'])
            X_train_survival, X_test_survival, y_train_survival, y_test_survival = train_test_split(new_df.drop(columns=['real_value', 'extreme_event', 'duration', 'start_date', 'stop_date']), 
                            Surv.from_dataframe(event='extreme_event', time='duration', data=new_df.drop(columns=['real_value'])), 
                            test_size=test_size_, shuffle=self.shuffle_, random_state=self.random_state)
            
            features_df = extract_features(data, self.timesteps)
            new_df_ext_feat_ = alter_duration_calculation(self.timesteps, data, columns=[self.Column_name, 'extreme_event'])
            new_df_ext_feat = new_df_ext_feat_.iloc[:, self.timesteps:]
            for i, column in enumerate(features_df.columns):
                new_df_ext_feat.insert(i, column, value=features_df[column])
            new_df_ext_feat.fillna(0, inplace=True)

            X_train_survival_ext_feat, X_test_survival_ext_feat, y_train_survival_ext_feat, y_test_survival_ext_feat =  train_test_split(new_df_ext_feat.drop(columns=['real_value', 'extreme_event', 'duration', 'start_date', 'stop_date']), 
                            Surv.from_dataframe(event='extreme_event', time='duration', data=new_df_ext_feat.drop(columns=['real_value'])), 
                            test_size=test_size_, shuffle=self.shuffle_, random_state=self.random_state)
            
            return X_train, X_test, y_train, y_test, THRESHOLD, X_train_survival, X_test_survival, y_train_survival, y_test_survival, X_train_survival_ext_feat, X_test_survival_ext_feat, y_train_survival_ext_feat, y_test_survival_ext_feat

    def invalid_observations_removal(self, X_train, X_test, y_train, y_test, THRESHOLD, invalid_observations_removal_):
        if invalid_observations_removal_:
            X_train, y_train = remove_invalid_observations(X=pd.DataFrame(X_train, columns=[f'lag_{i+1}' for i in range(self.timesteps)]),
                                                        y=y_train,
                                                        lag_columns=pd.DataFrame(X_train, columns=[f'lag_{i+1}' for i in range(self.timesteps)]).columns,
                                                        decision_thr=THRESHOLD)
            X_test, y_test = remove_invalid_observations(X=pd.DataFrame(X_test, columns=[f'lag_{i+1}' for i in range(self.timesteps)]),
                                                        y=y_test,
                                                        lag_columns=pd.DataFrame(X_test, columns=[f'lag_{i+1}' for i in range(self.timesteps)]).columns,
                                                        decision_thr=THRESHOLD)
        elif invalid_observations_removal_ == 'Only_from_train':
            X_train, y_train = remove_invalid_observations(X=pd.DataFrame(X_train, columns=[f'lag_{i+1}' for i in range(self.timesteps)]),
                                                        y=y_train,
                                                        lag_columns=pd.DataFrame(X_train, columns=[f'lag_{i+1}' for i in range(self.timesteps)]).columns,
                                                        decision_thr=THRESHOLD)

        return X_train, X_test, y_train, y_test
    

#TODO incorporate the feature extraction to the pipeline
def extract_features(df, timesteps):
    """
    Extract time series features from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing time series data.
    - timesteps (int): Number of time steps to consider for feature extraction.

    Returns:
    - features_df (pandas.DataFrame): DataFrame containing extracted time series features.

    Example Usage:
    timesteps = 10
    features_df = extract_features(df, timesteps)
    """

    # Initialize empty lists to store the extracted features
    mean_list = []
    std_list = []
    minimum_list = []
    maximum_list = []
    percentiles_a_list = []
    percentiles_b_list = []
    percentiles_c_list = []
    skewness_list = []
    kurtosis_list = []
    autocorr_list = []
    partial_autocorr_list = []
    power_spectrum_list = []
    rolling_mean_list = []
    rolling_std_list = []
    #real_values = []

    # Iterate over each step in the time series dataset
    for i in range(len(df)):
        # Extract features using data up to the current step
        data_up_to_current_step = df['Incidents'][:i+timesteps]

        # Compute statistical features
        mean = data_up_to_current_step.mean()
        std = data_up_to_current_step.std()
        minimum = data_up_to_current_step.min()
        maximum = data_up_to_current_step.max()
        percentiles_a = data_up_to_current_step.quantile(0.25)
        percentiles_b = data_up_to_current_step.quantile(0.5)
        percentiles_c = data_up_to_current_step.quantile(0.75)
        skewness = data_up_to_current_step.skew()
        kurtosis = data_up_to_current_step.kurtosis()
        #real_value = df['Incidents'].iloc[i]

        # Compute autocorrelation features
        autocorr = data_up_to_current_step.autocorr()
        #partial_autocorr = pd.Series(sm.tsa.stattools.pacf(data_up_to_current_step))

        # Compute frequency domain features
        fft = np.fft.fft(data_up_to_current_step)
        power_spectrum = np.abs(fft) ** 2

        # Compute time domain features
        #rolling_mean = data_up_to_current_step.rolling(window=i+1).mean()
        #rolling_std = data_up_to_current_step.rolling(window=i+1).std()

        # Append the extracted features to the respective feature lists
        mean_list.append(np.round(mean,2))
        std_list.append(np.round(std,2))
        minimum_list.append(np.round(minimum,2))
        maximum_list.append(np.round(maximum,2))
        percentiles_a_list.append(np.round(percentiles_a,2))
        percentiles_b_list.append(np.round(percentiles_b,2))
        percentiles_c_list.append(np.round(percentiles_c,2))
        skewness_list.append(np.round(skewness,2))
        kurtosis_list.append(np.round(kurtosis,2))
        autocorr_list.append(np.round(autocorr,2))
        #partial_autocorr_list.append(partial_autocorr)
        power_spectrum_list.append(power_spectrum)
        #rolling_mean_list.append(rolling_mean)
        #rolling_std_list.append(rolling_std)
        #real_values.append(real_value)

    # Create a new DataFrame for the extracted features
    features_df = pd.DataFrame({
        'Mean': mean_list,
        'Standard Deviation': std_list,
        'Minimum': minimum_list,
        'Maximum': maximum_list,
        'Percentiles_a': percentiles_a_list,
        'Percentiles_b': percentiles_b_list,
        'Percentiles_c': percentiles_c_list,
        'Skewness': skewness_list,
        'Kurtosis': kurtosis_list,
        'Autocorrelation': autocorr_list#,
        #'Partial Autocorrelation': partial_autocorr_list,
        #'Power Spectrum': power_spectrum_list,
        #'Rolling Mean': rolling_mean_list,
        #'Rolling Standard Deviation': rolling_std_list
        #'Real_value': real_values,
        #'Date': df['Date']
    })

    return features_df 
