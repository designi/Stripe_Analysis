# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:43:06 2020

@author: ngarcia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import warnings

#create predictions
def calcAvg(df,cntry):
    data = df[df.country==cntry].sort_values('date')
    data.set_index("date", inplace = True)
    avg_data = data.groupby('date').mean()
    return avg_data

def calcPrediction(df,countries,p=1,d=1,q=0,err_results=False):
    """
    Nonseasonal ARIMA model
    p = N autoregressive terms
    d = N differences for stationarity
    q = N lagged forecast errors 
    """
    datapoints = {}
    prediction_results = {}
    for cntry in countries:
        print(' Working on {}'.format(cntry))
        # dataset
        avg_data = calcAvg(df,cntry)
        datapoints[cntry] = len(avg_data)
        if len(avg_data) == 0: 
            prediction_results[cntry] = np.NaN
            print(' No data for {}. Investigate...'.format(cntry))
            continue
        model = ARIMA(avg_data,order=(p,d,q))
        # fit ARIMA model
        model_fit = model.fit(disp=False)
        # make prediction 
        yhat = model_fit.predict(start = len(avg_data), end=len(avg_data), typ='levels')
        # cntry-prediction map
        prediction_results[cntry] = float(yhat)
        def examineErrors(model_fit):
            residuals = pd.DataFrame(model_fit.resid)
            residuals.plot(title=cntry+' Residuals',legend=False)
            return 'Residuals {}\n'.format(residuals.describe())
        if err_results:
            print(examineErrors(model_fit))
    return prediction_results

def genPredictionGraph(prediction_results):
    assert isinstance(prediction_results,dict)
    if len(prediction_results) != 0:
        return plt.bar(x=prediction_results.keys(),height=prediction_results.values(),align='edge')

if __name__=='__main__':
    Verbose = False
    print(' Loading csv files...')
    df_industries = pd.read_csv(r'C:\Users\ngarcia\Downloads\industries_ngarcia.csv')
    df_payouts = pd.read_csv(r'C:\Users\ngarcia\Downloads\payouts_ngarcia.csv')
    df_countries = pd.read_csv(r'C:\Users\ngarcia\Downloads\countries_ngarcia.csv')
    
    # Ignore warnings
    warnings.filterwarnings("ignore")
    
    # Convert str to date
    df_payouts['date'] = pd.to_datetime(df_payouts['date']).dt.date
    
    print(' Estimating the amount of money expected to be paid out to each country on Jan. 1, 2019...')

    # Merge payouts/cntrys
    df_merge_pymnt_cntry = pd.merge(df_payouts,df_countries,left_on='recipient_id',right_on='merchant_id', how='inner')
    
    # Reassign for predictions
    df = df_merge_pymnt_cntry
    df.dropna(inplace=True)
    
    # Aggregate total payouts by date by country
    payout_by_date_cntry = df.groupby(['date','country']).sum()
    payout_by_date_cntry.sort_values(['date','country'])
    
    # Aggregate total payouts by country by date 
    payout_by_cntry_date = df.groupby(['country','date']).sum()
    payout_by_cntry_date.sort_values(['country','date'])
    
    # Drop count column
    df.drop(labels='count',inplace=True,axis=1)
    
    # Get country list
    countries=sorted(list(df.country.unique()))
    r = calcPrediction(df,countries,err_results=Verbose)
    print(' \n Finished calculating predictions...\n\n {} \n\n '.format(r))
    
    # Industry analysis
    df_merge_ind_cntry = pd.merge(df_payouts,df_industries,left_on='recipient_id',right_on='merchant_id', how='inner')
    df_merge_ind_cntry.sort_values(['industry','date'],inplace=True)
    ind = sorted(list(df_merge_ind_cntry.industry.unique()))
    
    # Map parameters
    paramdct = {'Education':15,'Hotels, Restaurants & Leisure':5,'Food & Beverage':40}
    
    df_merge_ind_cntry.groupby(['date','industry'])['amount'].sum()
    
    
    