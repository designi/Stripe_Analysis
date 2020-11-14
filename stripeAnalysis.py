# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:43:06 2020

@author: ngarcia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

#create predictions
def calc(df,cntry,plots=False):
    data = df[df.country==cntry].sort_values('date')
    data.set_index("date", inplace = True)
    results= data.groupby('date').sum()
    if not plots:
        return results
    results.plot(title='Amount paid out to {} daily'.format(cntry))
    if len(results) > 365/2:
        results.rolling(14).mean().plot(title='Bi-weekly rolling average payouts to {}'.format(cntry))
    

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
        data = calc(df,cntry)
        datapoints[cntry] = len(data)
        if len(data) == 0: 
            prediction_results[cntry] = np.NaN
            print(' No data for {}. Investigate...'.format(cntry))
            continue
        model = ARIMA(data,order=(p,d,q))
        # fit ARIMA model
        model_fit = model.fit(disp=False)
        # make prediction 
        yhat = model_fit.predict(start = len(data), end=len(data), typ='levels')
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

def genIndustryPredictions(df,industries,plots=False):
    prediction_results = {}
    for ind in industries:
        print(' Working on {}'.format(ind))
        df_ind = df[df.industry==ind].sort_values('date')
        df_ind.set_index("date", inplace = True)
        df_ind.index = pd.to_datetime(df_ind.index)
        tdv = df_ind['amount'].groupby('date').count()
        model = SARIMAX(tdv, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52),trend='t')
        # fit SARIMA model
        model_fit = model.fit(disp=False)
        # make prediction 
        yhat = model_fit.predict(start=len(tdv), end=len(tdv), typ='levels')
        prediction_results[ind] = int(yhat)
        if plots:
            tdv.plot(title=ind)
            tdv.diff()[1:].plot(title=ind+' lagged 1 period')
    return prediction_results
        
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

    # Drop count column
    df.drop(labels='count',inplace=True,axis=1)
    
    # Get country list
    countries=sorted(list(df.country.unique()))
    r = calcPrediction(df,countries,err_results=Verbose)
    print(' \n Finished calculating predictions...\n\n {} \n\n '.format(r))
    
    # Industry analysis
    df_merge_ind_payout = pd.merge(df_payouts,df_industries,left_on='recipient_id',right_on='merchant_id', how='inner')
    
    # Drop count column
    df_merge_ind_payout.drop(labels='count',inplace=True,axis=1)
    
    # Sort the df on ind and dt
    df_merge_ind_payout.sort_values(['industry','date'],inplace=True)
    industries = sorted(list(df_merge_ind_payout.industry.unique()))

    # Map parameters
    paramdct = {'Education':15,'Travel & Hospitality':5,'Food & Beverage':40}
    industries = list(paramdct.keys())
    
    #generate industry predictions
    res = genIndustryPredictions(df_merge_ind_payout,industries)
    print(' \n Finished calculating predictions...\n\n {} \n\n '.format(res))






