import json
import pandas as pd 
import numpy as np 
import requests
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import norm, laplace, logistic, gumbel_r, lognorm, cauchy, genextreme
import statsmodels.api as sm

def data_request(apikey, headers, search_endpoint, data_dir, ingredient, hazard, interval, starting_date, end_date, plot=True):
    """
    Request and process data related to incidents for a specific ingredient and hazard.

    Parameters:
    - apikey (str): API key for the data request.
    - headers (dict): Request headers.
    - search_endpoint (str): Endpoint for the data search.
    - data_dir (str): Directory for data storage.
    - ingredient (str): Ingredient value.
    - hazard (str): Hazard value.
    - interval (str): Interval for data aggregation.
    - starting_date (str): Start date for data retrieval.
    - end_date (str): End date for data retrieval.
    - plot (bool, optional): Whether to plot the data. Defaults to True.

    Returns:
    - data_ (pandas.DataFrame): Processed data in a DataFrame.

    Example Usage:
    apikey = 'your_api_key'
    headers = {'Content-Type': 'application/json'}
    search_endpoint = 'https://example.com/api/search'
    data_dir = '/data'
    ingredient = 'example_ingredient'
    hazard = 'example_hazard'
    interval = 'month'
    starting_date = '2023-01-01'
    end_date = '2023-12-31'
    plot_data = True
    data_df = data_request(apikey, headers, search_endpoint, data_dir, ingredient, hazard, interval, starting_date, end_date, plot_data)
    """

    base_request = {
        'apikey': apikey,
        'detail': True,
        'entityType': 'incident',
        'pageSize': 0, #or use 1 here
        'published': True,
        'from': starting_date,
        'to': end_date,
        'strictQuery': {
                'products.value': ingredient, 
                'hazards.value': hazard,
                #'dataSource': "!!USDA&&!!FDA&&!!Food Safety Authority in China",
                #"origin.value": "europe"
            },
        'aggregations': {
            "date_distribution": {  # distribution of incidents over the years
                "attribute": "createdOn",
                "interval": interval,
                "format": "YYYY-MM",
                "size": 1000
            }
        }
    }

    data = requests.post(search_endpoint, headers=headers, data=json.dumps(base_request)).json()
    #print(json.dumps(data, indent=4))

    dataset = {'Date':[], 'Incidents':[]}

    for i in json.loads(json.dumps(data, indent=4))['aggregations']['date_histogram#date_distribution']['buckets']:
        date = i['key_as_string']
        num = i['doc_count']
        dataset['Date'].append(date)
        dataset['Incidents'].append(num)


    data_ = pd.DataFrame(dataset, columns = ['Date', 'Incidents'])

    #data_.to_csv('/home/gmarinos/Documents/Code/1st_publication/experiments/data/salmonella_incidents.txt',index=None, mode='a')

    #data['Date'] = pd.to_datetime(data_['Date'])

    def plot_data(data_):

        data_ = data_.set_index('Date')
        #data_.to_csv('predictions_.csv')

        print('Incidents', data_['Incidents'].sum())

        id = [data_.index[i] for i in range(0, len(data_.index), 20)]

        plt.figure(figsize=(12,6))
        plt.plot(data_.Incidents, color='black')
        plt.title('Number of '+str(hazard)+' '+str('hazard')+' '+str('ingredient:')+' '+str(ingredient)+' during years')
        plt.xticks(ticks=id, rotation=45)
        #plt.xticks(rotation=45)
        #print(data_.Incidents.iloc[-1])
        #plt.hlines(y=320, xmin=data_.index[0], xmax=data_.index[-1], color='red', linestyle='--')
        plt.show()

        if plot == True:
            plot_data(data_ = data_)

    return data_


def data_prediction_request(apikey, headers, search_endpoint, data_dir, ingredient, hazard, interval, starting_date, end_date):
    """
    Request and process prediction trend data from an API.

    Parameters:
    - apikey (str): API key for authentication.
    - headers (dict): Request headers.
    - search_endpoint (str): API endpoint for data retrieval.
    - data_dir (str): Directory to save the data.
    - ingredient (str): Ingredient name for the query.
    - hazard (str): Hazard type for the query.
    - interval (str): Time interval for data aggregation (e.g., 'month', 'year').
    - starting_date (str): Start date for the data query (e.g., 'YYYY-MM-DD').
    - end_date (str): End date for the data query (e.g., 'YYYY-MM-DD').

    Returns:
    - data_ (pandas.DataFrame): Processed DataFrame containing prediction trend data.

    Example Usage:
    apikey = 'your_api_key'
    headers = {'Content-Type': 'application/json'}
    search_endpoint = 'https://api.example.com/search'
    data_dir = '/path/to/data/directory'
    ingredient = 'Salmonella'
    hazard = 'Foodborne Illness'
    interval = 'month'
    starting_date = '2023-01-01'
    end_date = '2023-12-31'
    df = data_prediction_request(apikey, headers, search_endpoint, data_dir, ingredient, hazard, interval, starting_date, end_date)
    """

    base_request = {
        'apikey': apikey,
        'detail': True,
        'entityType': 'prediction_trend',
        'pageSize': 0, #or use 1 here
        'published': True,
        'from': starting_date,
        'to': end_date,
        'strictQuery': {
                'information.product': ingredient,
                'information.hazard': hazard,
                #'information.dataSource': "!!USDA&&!!FDA&&!!Food Safety Authority in China"#,
                "information.origin": "NONE",
                "information.interval": interval
            },
        'aggregations': {
                "date_distribution": { # distribution of incidents over the years
                "attribute": "createdOn",
                "interval": interval,
                "format": "YYYY-MM",
                "size": 1000,

                "subAggregation": {
                    "attribute": "information.actual_value",
                    "sum": True
                }
            }
        }
    }

    data = requests.post(search_endpoint, headers=headers, data=json.dumps(base_request)).json()
    #print(json.dumps(data, indent=4))

    dataset = {'Date':[], 'Predicted_value':[]}

    for i in json.loads(json.dumps(data, indent=4))['aggregations']['date_histogram#date_distribution']['buckets']:
        date = i['key_as_string']
        num = i["sum#subaggregation"]['value']
        dataset['Date'].append(date)
        dataset['Predicted_value'].append(num)


    data_ = pd.DataFrame(dataset, columns = ['Date', 'Predicted_value'])


    #data_.to_csv('/home/gmarinos/Documents/Code/1st_publication/experiments/data/salmonella_incidents.txt',index=None, mode='a')

    #data['Date'] = pd.to_datetime(data_['Date'])

    data_['Date'] = pd.to_datetime(data_['Date'])
    data_ = data_.set_index('Date')

    return data_