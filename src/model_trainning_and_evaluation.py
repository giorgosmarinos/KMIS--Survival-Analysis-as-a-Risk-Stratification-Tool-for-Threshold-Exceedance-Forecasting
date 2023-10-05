import sklearn 
import pickle 
import numpy as np
import pandas as pd
import io 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from scipy.stats import norm, laplace, logistic, gumbel_r, lognorm, cauchy, genextreme 
from hmmlearn import hmm
from sksurv.ensemble import ExtraSurvivalTrees, RandomSurvivalForest
from sksurv.util import Surv
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from tabulate import tabulate
from src.synthetic_data_generation.synthetic_data_generation import *
from src.utils.utils import *
from src.data_fetching.data_fetching import *
from src.dataset_construction.dataset_construction import *

class ModelEvaluator:

    """
    A versatile class for evaluating machine learning models and statistical techniques on a given dataset.

    Attributes:
        data (pd.DataFrame): The dataset for evaluation.
        X_train, X_test, y_train, y_test (pd.DataFrame): Train and test data splits.
        THRESHOLD (float): A threshold value used in evaluation.
        X_train_survival, X_test_survival, y_train_survival, y_test_survival (pd.DataFrame): Train and test data splits for survival analysis.
        X_train_survival_ext_feat, X_test_survival_ext_feat, y_train_survival_ext_feat, y_test_survival_ext_feat (pd.DataFrame):
            Train and test data splits for survival analysis with extracted features.
        target_column (str): The target column for evaluation.
        timesteps (int): Number of timesteps or lags considered in analysis.
        THRESHOLD_PERCENTILE (float): A percentile threshold used in evaluation.
        validation_rows (int): Number of rows used for validation.
        random_state (int): Random seed for reproducibility.
        test_size_ (float): Size of the test dataset.
        shuffle_ (bool): Whether to shuffle the data during evaluation.
        invalid_observations_removal (bool): Whether to remove invalid observations.
        ingredient (str): A descriptor for the dataset.
        Column_name (str): A descriptor for the dataset column.
        interval (str): A descriptor for the time interval considered in analysis.

    Methods:
        evaluate_models():
            Loop through various models and distributions, fit each model, and compute performance metrics including accuracy,
            precision, recall, F1-score, and ROC AUC. Handles specific logic for certain models like Prophet and survival analysis.

        print_results():
            Print the evaluation results in a tabular format, including data details, thresholds, model names, and performance
            metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

        get_results_df():
            Return the evaluation results as a DataFrame and save the results to CSV files for further analysis.
    """
    
    def __init__(self, data, 
                 X_train, X_test, y_train, y_test, THRESHOLD, X_train_survival, X_test_survival, y_train_survival, y_test_survival, 
                 X_train_survival_ext_feat, X_test_survival_ext_feat, y_train_survival_ext_feat, y_test_survival_ext_feat,
                 target_column, timesteps, THRESHOLD_PERCENTILE,
                 validation_rows, random_state, test_size_, shuffle_, invalid_observations_removal, Column_name, interval, data_ingredient, predictions_ingredient, data_hazard, predictions_hazard):
        self.data = data
        self.X_train = X_train 
        self.X_test = X_test 
        self.y_train = y_train 
        self.y_test = y_test 
        self.THRESHOLD = THRESHOLD 
        self.X_train_survival = X_train_survival 
        self.X_test_survival = X_test_survival 
        self.y_train_survival = y_train_survival
        self.y_test_survival = y_test_survival
        self.X_train_survival_ext_feat = X_train_survival_ext_feat
        self.X_test_survival_ext_feat = X_test_survival_ext_feat
        self.y_train_survival_ext_feat = y_train_survival_ext_feat
        self.y_test_survival_ext_feat = y_test_survival_ext_feat
        self.target_column = target_column
        self.timesteps = timesteps
        self.THRESHOLD_PERCENTILE = THRESHOLD_PERCENTILE
        self.validation_rows = validation_rows
        self.random_state = random_state
        self.test_size_ = test_size_
        self.shuffle_ = shuffle_
        self.invalid_observations_removal = invalid_observations_removal
        self.Column_name = Column_name
        self.interval = interval
        self.list_of_models = [DecisionTreeRegressor(random_state=random_state), 
                               RandomForestRegressor(random_state=random_state), 
                               SGDRegressor(random_state=random_state), 
                               KNeighborsRegressor(n_neighbors=3), 
                               AdaBoostRegressor(random_state=random_state),
                               BayesianRidge(), 
                               ExtraTreesRegressor(random_state=random_state),
                               BaggingRegressor(random_state=random_state), 
                               LGBMRegressor(objective="regression",random_state=random_state),
                               SVR(kernel="linear", C=0.025),
                               SVR(gamma=2, C=1),
                               #GaussianProcessRegressor(1.0 * RBF(1.0), random_state=random_state),
                               MLPRegressor(random_state=random_state),
                               'Prophet',
                               'hidden_markov_models',
                               'Survival_Analysis',
                               'Survival_Analysis_extracted_features',
                               'Prophet_simple_threshold'
                              ]

        self.distributions = [genextreme, norm, laplace, logistic, gumbel_r, lognorm, cauchy]
        self.table_data = []
        self.trained_regressors = {}
        self.results = pd.DataFrame(index=self.y_test.index)
        self.data_ingredient = data_ingredient
        self.predictions_ingredient = predictions_ingredient
        self.data_hazard = data_hazard
        self.predictions_hazard = predictions_hazard

        list_of_choices = {'data_ingredient':self.data_ingredient, 
                    'data_hazard':self.data_hazard, 
                    'predictions_ingredient':self.predictions_ingredient, 
                    'predictions_hazard':self.predictions_hazard}

        keys_to_remove = []  # Create a list to store keys to remove

        for key, value in list_of_choices.items():
            if value == None:
                keys_to_remove.append(key)

            if value == "NONE":
                keys_to_remove.append(key)

            if value == 'None':
                keys_to_remove.append(key)

        # Iterate over the keys to remove and delete corresponding items from the dictionary
        for key in keys_to_remove:
            del list_of_choices[key]


        self.ingredient = ''

        for key, value in list_of_choices.items():
            self.ingredient += str(key)+':'+str(value)+' _ '

        


    def evaluate_models(self):

        # Loop through models and distributions and evaluate each one
        for regression in self.list_of_models:
            for distribution in self.distributions:
                if regression not in ['Prophet', 'Prophet_simple_threshold', 'Survival_Analysis', 'hidden_markov_models', 'Survival_Analysis_extracted_features']:
                    # Implement the model fitting and evaluation logic here
                    print('\n')
                    print('Distribution:', distribution.name)
                    print('Model:', str(regression))

                    regression.fit(self.X_train, self.y_train)

                    self.trained_regressors[str(regression)+'_'+str(distribution.name)]=[regression]

                    # getting point forecasts
                    point_forecasts = regression.predict(self.X_test)
                    std = self.y_train.std()
                    if distribution == lognorm:
                        exceedance_prob = np.asarray([1 - distribution.cdf(self.THRESHOLD, loc=x_, s=std) for x_ in point_forecasts])
                    elif distribution == genextreme:
                        exceedance_prob = np.asarray([1 - distribution.cdf(self.THRESHOLD, c = -0.1, loc=x_, scale=std) for x_ in point_forecasts])
                    elif distribution != lognorm:
                        exceedance_prob = np.asarray([1 - distribution.cdf(self.THRESHOLD, loc=x_, scale=std) for x_ in point_forecasts])
                    
                    #y_test_ = new_df['marked'].loc[y_test.index]
                    y_test_ = (self.y_test > self.THRESHOLD).astype(int)
                    
                    y_pred_binary = np.where(exceedance_prob >= 0.5, 1, 0)
                    self.results['actual_predictions_from_model'+'_'+str(regression)+'_'+str(distribution.name)]  = point_forecasts
                    self.results['y_pred_binary'+'_'+str(regression)+'_'+str(distribution.name)]  = y_pred_binary
                    self.trained_regressors[str(regression)+'_'+str(distribution.name)].append(y_pred_binary)

                    # Compute the confusion matrix
                    cm = confusion_matrix(y_test_, y_pred_binary)

                    # Compute the precision, recall, and F1 score
                    accuracy = accuracy_score(y_test_, y_pred_binary)
                    precision = precision_score(y_test_, y_pred_binary)
                    recall = recall_score(y_test_, y_pred_binary)
                    f1 = f1_score(y_test_, y_pred_binary)
                    roc_auc = roc_auc_score(y_test_, exceedance_prob)

                    # Print the results
                    print(f"Confusion Matrix:\n{cm}")
                    print(f"Accuracy: {accuracy:.2f}")
                    print(f"Precision: {precision:.2f}")
                    print(f"Recall: {recall:.2f}")
                    print(f"F1 Score: {f1:.2f}")
                    print(f"Roc AUC: {roc_auc:.2f}")

                    self.trained_regressors[str(regression)+'_'+str(distribution.name)].append([accuracy, precision, recall, f1, roc_auc])

                    self.table_data.append([self.ingredient+'_'+self.Column_name, self.THRESHOLD, self.timesteps, distribution.name, regression, 
                                        str(int(np.round(accuracy, 2)*100))+str('%'), 
                                        str(int(np.round(precision, 2)*100))+str('%'), 
                                        str(int(np.round(recall, 2)*100))+str('%'), 
                                        str(int(np.round(f1, 2)*100))+str('%'), 
                                        str(int(np.round(roc_auc, 2)*100))+str('%'), 
                                        self.test_size_, self.shuffle_, self.invalid_observations_removal, 
                                        len(self.X_train.columns), 
                                        self.interval])
      
                elif regression == 'Prophet' and self.test_size_ == 0:
                    # Implement Prophet model logic here
                    self.trained_regressors[str(regression)+'_'+str(distribution.name)]=[regression]

                    prophet_predictions = data_prediction_request(self.apikey, self.headers, self.search_endpoint, self.data_dir, self.predictions_ingredient, 
                                                                  self.predictions_hazard, self.interval, self.starting_date, self.end_date)
                    predicted_historical_values = prophet_predictions[:self.validation_rows].values

                    #------------------------------------------
                    #list_of_results = insert_metrics_from_foodakai(predicted_historical_values, y_train, y_test, exceedance_prob, Column_name, 
                    #                       THRESHOLD, timesteps, foodakai_model='Prophet',
                    #                      test_size=test_size_, shuffle=shuffle_, invalid_observations_removal = invalid_observations_removal, 
                    #                      no_of_columns= len(X_train.columns))

                    #for i in list_of_results:
                    #  table_data.append(i)
                    #--------------------------------------------

                    std = self.y_train.std()
                    if distribution == lognorm:
                        exceedance_prob = np.asarray([1 - distribution.cdf(self.THRESHOLD, loc=x_, s=std) for x_ in predicted_historical_values])
                    elif distribution == genextreme:
                        exceedance_prob = np.asarray([1 - distribution.cdf(self.THRESHOLD, c = -0.1, loc=x_, scale=std) for x_ in predicted_historical_values])
                    elif distribution != lognorm:
                        exceedance_prob = np.asarray([1 - distribution.cdf(self.THRESHOLD, loc=x_, scale=std) for x_ in predicted_historical_values])

                    y_test_ = (self.y_test > self.THRESHOLD).astype(int)
                    y_pred_binary = np.where(exceedance_prob >= 0.5, 1, 0)

                    self.trained_regressors[str(regression)+'_'+str(distribution.name)].append(y_pred_binary)

                    self.results['actual_predictions_from_model'+'_'+str(regression)]  = predicted_historical_values
                    self.results['y_pred_binary'+'_'+str(regression)+'_'+str(distribution.name)]  = y_pred_binary

                    # Calculate metrics
                    accuracy = accuracy_score(y_test_, y_pred_binary)
                    precision = precision_score(y_test_, y_pred_binary)
                    recall = recall_score(y_test_, y_pred_binary)
                    f1 = f1_score(y_test_, y_pred_binary)
                    roc_auc = roc_auc_score(y_test_, exceedance_prob)

                    self.table_data.append([self.ingredient+'_'+self.Column_name, self.THRESHOLD, self.timesteps, distribution.name, regression, 
                                        str(int(np.round(accuracy, 2)*100))+str('%'), 
                                        str(int(np.round(precision, 2)*100))+str('%'), 
                                        str(int(np.round(recall, 2)*100))+str('%'), 
                                        str(int(np.round(f1, 2)*100))+str('%'), 
                                        str(int(np.round(roc_auc, 2)*100))+str('%'), 
                                        self.test_size_, self.shuffle_, self.invalid_observations_removal, 
                                        len(self.X_train.columns), 
                                        self.interval])

                    self.trained_regressors[str(regression)+'_'+str(distribution.name)].append([accuracy, precision, recall, f1, roc_auc])

                else:
                    pass

            if regression == 'hidden_markov_models' and self.test_size_ == 0:
                # Implement HMM model logic here
                print('\n')
                print('Model:', str(regression))

                self.trained_regressors[str(regression)]=[regression]

                # Initialize the HMM model
                model = hmm.CategoricalHMM(n_components=2, random_state=33)

                observed_window = self.data['Incidents'][:-self.validation_rows]

                # Encode the observed values based on the threshold
                binary_sequence = np.array(observed_window > self.THRESHOLD, dtype=int)

                # Preprocess the data to handle missing or invalid values
                binary_sequence[np.isnan(binary_sequence)] = 0  # Replace NaN values with 0

                # Fit the model to the current observed window
                model.fit(binary_sequence.reshape(-1, 1))

                # Forecast the next hidden state
                num_steps = len(self.data[-self.validation_rows:])  # Number of future steps to forecast
                future_hidden_states, _ = model.sample(num_steps)

                y_test_ = (self.y_test > self.THRESHOLD).astype(int)

                y_pred_binary = future_hidden_states

                # Compute the confusion matrix
                cm = confusion_matrix(y_test_, y_pred_binary)

                # Calculate metrics
                accuracy = accuracy_score(y_test_, y_pred_binary)
                precision = precision_score(y_test_, y_pred_binary)
                recall = recall_score(y_test_, y_pred_binary)
                f1 = f1_score(y_test_, y_pred_binary)
                roc_auc = roc_auc_score(y_test_, y_pred_binary)

                # Print the results
                print(f"Confusion Matrix:\n{cm}")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1 Score: {f1:.2f}")
                print(f"Roc AUC: {roc_auc:.2f}")

                self.trained_regressors[str(regression)].append(y_pred_binary)

                self.results['actual_predictions_from_model'+'_'+str(regression)]  = predicted_historical_values
                self.results['y_pred_binary'+'_'+str(regression)]  = y_pred_binary

                self.table_data.append([self.ingredient+'_'+self.Column_name, self.THRESHOLD, self.timesteps, None, regression, 
                                    str(int(np.round(accuracy, 2)*100))+str('%'), 
                                    str(int(np.round(precision, 2)*100))+str('%'), 
                                    str(int(np.round(recall, 2)*100))+str('%'), 
                                    str(int(np.round(f1, 2)*100))+str('%'), 
                                    str(int(np.round(roc_auc, 2)*100))+str('%'), 
                                    self.test_size_, self.shuffle_, self.invalid_observations_removal, 
                                    len(self.X_train.columns), 
                                    self.interval])

                self.trained_regressors[str(regression)].append([accuracy, precision, recall, f1, roc_auc])
                


            if regression == 'Survival_Analysis':
                # Implement Survival Analysis logic here
                self.trained_regressors[str(regression)]=[regression]

                scores_cph_tree = {}
                scores_RandomSurvivalForest = {} 

                est_cph_tree = RandomSurvivalForest(random_state=self.random_state)
                for i in range(1, 61):
                    n_estimators = i * 5
                    est_cph_tree.set_params(n_estimators=n_estimators)
                    est_cph_tree.fit(self.X_train_survival, self.y_train_survival)
                    scores_cph_tree[n_estimators] = est_cph_tree.score(self.X_test_survival, self.y_test_survival)
                    scores_RandomSurvivalForest['scores_cph_tree'+str(n_estimators)] =  pickle.dumps(est_cph_tree)
                

                metrics = []
                prob_percentages = [0.85, 0.90]

                for i in range(5, 61, 5):
                    for prob_percentage in prob_percentages:
                        list_of_lists = pickle.loads(scores_RandomSurvivalForest['scores_cph_tree'+str(i)]).predict_survival_function(self.X_test_survival, return_array=True)

                        y_test_ = (self.y_test_survival['extreme_event'] == True).astype(int)
                        y_pred_binary = pd.DataFrame(list_of_lists[0:,0], columns=['probabilities'])['probabilities'].apply(lambda x: 1 if x < prob_percentage else 0).values

                        # Calculate metrics
                        accuracy = accuracy_score(y_test_, y_pred_binary)
                        precision = precision_score(y_test_, y_pred_binary)
                        recall = recall_score(y_test_, y_pred_binary)
                        f1 = f1_score(y_test_, y_pred_binary)
                        roc_auc = roc_auc_score(y_test_, list_of_lists[0:,0])
                        

                        metrics.append({'model':'scores_cph_tree'+str(i),
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'roc_auc':roc_auc,
                                        'f1-score':f1,
                                        'y_pred_binary':y_pred_binary})

                        print(accuracy,'%', precision,'%', recall,'%', f1,'%', roc_auc,'%')

                # Sort the list based on the 'age' field
                sorted_data = sorted(metrics, key=lambda x: x['f1-score'], reverse=True)

                accuracy = sorted_data[0]['accuracy']
                precision = sorted_data[0]['precision']
                recall = sorted_data[0]['recall']    
                f1 = sorted_data[0]['f1-score']
                roc_auc = sorted_data[0]['roc_auc']

                self.trained_regressors[str(regression)].append(y_pred_binary)
                
                self.table_data.append([self.ingredient+'_'+self.Column_name, self.THRESHOLD, self.timesteps, None, 
                                    regression+'_'+str(sorted_data[0]['model']), 
                                    str(int(np.round(accuracy,2)*100))+str('%'), 
                                    str(int(np.round(precision,2)*100))+str('%'), 
                                    str(int(np.round(recall,2)*100))+str('%'), 
                                    str(int(np.round(f1,2)*100))+str('%'), 
                                    str(int(np.round(roc_auc,2)*100))+str('%'), 
                                    self.test_size_, self.shuffle_, self.invalid_observations_removal, 
                                    len(self.X_train_survival.columns), 
                                    self.interval])

                
                self.trained_regressors[str(regression)].append([accuracy, precision, recall, f1, roc_auc])


            if regression == 'Survival_Analysis_extracted_features':
                # Implement Survival Analysis with extracted features logic here
                self.trained_regressors[str(regression)]=[regression]

                scores_cph_tree = {}
                scores_RandomSurvivalForest = {} 

                est_cph_tree = RandomSurvivalForest(random_state=self.random_state)
                for i in range(1, 61):
                    n_estimators = i * 5
                    est_cph_tree.set_params(n_estimators=n_estimators)
                    est_cph_tree.fit(self.X_train_survival_ext_feat, self.y_train_survival_ext_feat)
                    scores_cph_tree[n_estimators] = est_cph_tree.score(self.X_test_survival_ext_feat, self.y_test_survival_ext_feat)
                    scores_RandomSurvivalForest['scores_cph_tree'+str(n_estimators)] =  pickle.dumps(est_cph_tree)
                

                metrics = []
                prob_percentages = [0.85, 0.90]

                for i in range(5, 61, 5):
                    for prob_percentage in prob_percentages:
                        list_of_lists = pickle.loads(scores_RandomSurvivalForest['scores_cph_tree'+str(i)]).predict_survival_function(self.X_test_survival_ext_feat, return_array=True)

                        y_test_ = (self.y_test_survival_ext_feat['extreme_event'] == True).astype(int)
                        y_pred_binary = pd.DataFrame(list_of_lists[0:,8], columns=['probabilities'])['probabilities'].apply(lambda x: 1 if x < prob_percentage else 0).values

                        # Calculate metrics
                        accuracy = accuracy_score(y_test_, y_pred_binary)
                        precision = precision_score(y_test_, y_pred_binary)
                        recall = recall_score(y_test_, y_pred_binary)
                        f1 = f1_score(y_test_, y_pred_binary)
                        roc_auc = roc_auc_score(y_test_, list_of_lists[0:,8])
                        

                        metrics.append({'model':'scores_cph_tree'+str(i),
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'roc_auc':roc_auc,
                                        'f1-score':f1,
                                        'y_pred_binary':y_pred_binary})

                        print(accuracy,'%', precision,'%', recall,'%', f1,'%', roc_auc,'%')

                # Sort the list based on the 'age' field
                sorted_data = sorted(metrics, key=lambda x: x['f1-score'], reverse=True)

                accuracy = sorted_data[0]['accuracy']
                precision = sorted_data[0]['precision']
                recall = sorted_data[0]['recall']    
                f1 = sorted_data[0]['f1-score']
                roc_auc = sorted_data[0]['roc_auc']

                self.trained_regressors[str(regression)].append(y_pred_binary)
                
                self.table_data.append([self.ingredient+'_'+self.Column_name, self.THRESHOLD, self.timesteps, None, 
                                    regression+'_'+str(sorted_data[0]['model']), 
                                    str(int(np.round(accuracy,2)*100))+str('%'), 
                                    str(int(np.round(precision,2)*100))+str('%'), 
                                    str(int(np.round(recall,2)*100))+str('%'), 
                                    str(int(np.round(f1,2)*100))+str('%'), 
                                    str(int(np.round(roc_auc,2)*100))+str('%'), 
                                    self.test_size_, self.shuffle_, self.invalid_observations_removal, 
                                    len(self.X_train_survival_ext_feat.columns), 
                                    self.interval])

                
                self.trained_regressors[str(regression)].append([accuracy, precision, recall, f1, roc_auc])




            if regression == 'Prophet_simple_threshold' and self.test_size_ == 0:
                # Implement Prophet model with threshold logic here
                self.trained_regressors[str(regression)]=[regression]

                #-----------------------------
                #results_ = insert_metrics_from_foodakai_simple_threshold(predicted_historical_values, y_train, y_test, exceedance_prob, 
                #                       Column_name, THRESHOLD, timesteps, foodakai_model='Prophet',
                #                      test_size=test_size_, shuffle=shuffle_, invalid_observations_removal = invalid_observations_removal, 
                #                      no_of_columns= len(X_train.columns))

                #table_data.append(results_)
                #-----------------------------

                y_test_ = (self.y_test > self.THRESHOLD).astype(int)
                y_pred_binary = (predicted_historical_values > self.THRESHOLD).astype(int)
                
                self.trained_regressors[str(regression)].append(y_pred_binary)

                self.results['actual_predictions_from_model'+'_'+str(regression)]  = predicted_historical_values
                self.results['y_pred_binary'+'_'+str(regression)]  = y_pred_binary

                # Calculate metrics
                accuracy = accuracy_score(y_test_, y_pred_binary)
                precision = precision_score(y_test_, y_pred_binary)
                recall = recall_score(y_test_, y_pred_binary)
                f1 = f1_score(y_test_, y_pred_binary)
                roc_auc = roc_auc_score(y_test_, y_pred_binary)

                self.table_data.append([self.ingredient+'_'+self.Column_name, self.THRESHOLD, self.timesteps, None, regression, 
                                    str(int(np.round(accuracy, 2)*100))+str('%'), 
                                    str(int(np.round(precision, 2)*100))+str('%'), 
                                    str(int(np.round(recall, 2)*100))+str('%'), 
                                    str(int(np.round(f1, 2)*100))+str('%'), 
                                    str(int(np.round(roc_auc, 2)*100))+str('%'), 
                                    self.test_size_, self.shuffle_, self.invalid_observations_removal, 
                                    len(self.X_train.columns), 
                                    self.interval])

                
                self.trained_regressors[str(regression)].append([accuracy, precision, recall, f1, roc_auc])

        
    
    
    def print_results(self):
        # Print results and display the table
        print(tabulate(self.table_data, headers=["Data", "Threshold", "Number of lags", "Distribution", "Algorithm", 
                                    "Accuracy", "Precision", "Recall", "F1-score", 'Roc AUC', 'Test size', 'Shuffle', 
                                    'Invalid Observations removal', 'No of Columns', 'Resolution'], tablefmt="grid"))
        
        self.table_str = tabulate(self.table_data, headers=["Data", "Threshold", "Number of lags", "Distribution", "Algorithm", 
                                    "Accuracy", "Precision", "Recall", "F1-score", 'Roc AUC', 'Test size', 'Shuffle', 
                                    'Invalid Observations removal', 'No of Columns', 'Resolution'], tablefmt="pipe")
        return self.table_str

    def get_results_df(self):
        # Return the results dataframe
        df = pd.read_csv(io.StringIO(self.table_str), sep="|")
        df.to_csv('results/performance_results/performance_results'+'_'+self.ingredient+'_'+'threshold'+'_'+str(np.round(self.THRESHOLD,0))+'.'+'csv', index=False)
        self.results.to_csv('results/combined_results/results'+'_'+self.ingredient+'_'+'threshold'+'_'+str(np.round(self.THRESHOLD,0))+'.'+'csv')