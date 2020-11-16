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
    if plots:
        results.plot(title='Amount paid out to {} daily'.format(cntry))
        if len(results) > 365/2:
            results.rolling(14).mean().plot(title='Bi-weekly rolling average payouts to {}'.format(cntry))
    return results


def calcPrediction(df,countries,p=1,d=1,q=0,err_results=False):
    """
    Nonseasonal ARIMA model
    p = N autoregressive terms
    d = N differences for stationarity
    q = N lagged forecast errors 
    """
    datapoints = {}
    prediction_results = {}
    print(' Estimating the amount of money expected to be paid out to each country on Jan. 1, 2019...')
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
    print(' \n Finished calculating predictions...\n\n')
    return prediction_results

def genPredictionGraph(prediction_results):
    assert isinstance(prediction_results,dict)
    if len(prediction_results) != 0:
        return plt.bar(x=prediction_results.keys(),height=prediction_results.values(),align='edge')

def genIndustryPredictions(df,industries,merchant_cnt,sarima=False,plots=False):
    print(' \n Estimating total payout volume for {0}...\n\n'.format(industries))
    results = {}
    for ind in industries:
        print(' Working on {}'.format(ind))
        df_ind = df[df.industry==ind].sort_values('date')
        df_ind.set_index("date", inplace = True)
        df_ind.index = pd.to_datetime(df_ind.index)
        tdv = df_ind['amount'].groupby('date').count()
        if sarima:
            model = SARIMAX(tdv, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52),trend='t')
            # fit SARIMA model
            model_fit = model.fit(disp=False)
            # make prediction 
            yhat = model_fit.predict(start=len(tdv), end=len(tdv), typ='levels')
            results[ind] = int(yhat)
            if plots:
                tdv.plot(title=ind)
                tdv.diff()[1:].plot(title=ind+' lagged 7 periods')
            continue
        subset = df_merge_ind_payout[df_merge_ind_payout['industry']==ind]
        merchants = subset.groupby(['date','merchant_id'],as_index=False).count().groupby('date').count()
        merchants_by_day = pd.DataFrame(merchants['merchant_id'])
        tdv = pd.DataFrame(tdv)
        tdv.columns = ['transactions']
        df_ind = df_merge_ind_payout[df_merge_ind_payout.industry==ind].sort_values('date')
        merged_results = pd.merge(tdv,merchants_by_day,left_index=True, right_index=True)
        merged_results['average_by_merchant'] = merged_results['transactions']/merged_results['merchant_id']
        total_avg = np.mean(merged_results['average_by_merchant'])     
        coef = merchant_cnt.get(ind)
        print(' Multiplying average by merchant ({0}) by provided coefficient ({1}) to calculate Total Payout Volume'.format(total_avg,coef))
        results[ind] = total_avg * coef
    return results

def getDayOfWeekTrend(df,industries):
    ind_subset = getIndSubset(df,industries)
    dateDict = {0: 'Monday',1: 'Tuesday', 2: 'Wednesday',3: 'Thursday',4: 'Friday', 5: 'Saturday', 6: 'Sunday'}    
    ind_group_dict = {}
    for ind in industries:
        ind_dow = ind_subset.where(ind_subset.industry==ind)
        ind_dow.drop(columns='industry',inplace=True)
        ind_dow.dropna(axis = 0,inplace=True)
        ind_dow['Day_of_week'] = pd.to_datetime(ind_dow['date']).dt.dayofweek 
        j = ind_dow[['date','Day_of_week']].groupby('Day_of_week').count()
        j.index = j.index.map(dateDict.get)
        j.columns = ['Total_volume_by_day']
        plt = j.plot(kind='bar',title=ind+' Total Volume by Day',legend=False,rot=50)
        plt.set(ylabel='Total Volume')
        ind_group_dict[ind] = j
    return ind_group_dict

def getIndSubset(df,industries):
    ind_subset = df[df.industry.isin(industries)][['date','industry','amount']]
    ind_subset['date'] = pd.to_datetime(ind_subset['date'])
    return ind_subset


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
    print(r)
    
    # Industry analysis
    df_merge_ind_payout = pd.merge(df_payouts,df_industries,left_on='platform_id',right_on='merchant_id', how='inner')
    
    # Drop count column
    df_merge_ind_payout.drop(labels='count',inplace=True,axis=1)
    
    # Sort the df on ind and dt
    df_merge_ind_payout.sort_values(['industry','date'],inplace=True)
    df_merge_ind_payout.dropna(axis = 0,inplace=True)

    # Map parameters
    merchant_cnt = {'Education':15,'Travel & Hospitality':5,'Food & Beverage':40}
    industries = list(merchant_cnt.keys())

    #generate Sarima industry predictions
    res = genIndustryPredictions(df_merge_ind_payout,industries,merchant_cnt)
    print(res)
        

        
