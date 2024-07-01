"""
This script contains functions for finding:
    - The regularity of soldier movement with entropy and autoregressive modeling
    - The variability of kinematic measures over time with entropy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import EntropyHub as EH
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
# from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error



def PACS_entropy(ruck_slices_oriented, names):
    """
    Get entropy from PACS locations
    Using heatmaps and their shannon entropy

    Args:
        ruck_slices_oriented (list): DataFrames for each movement period
        names (list): list of soldier names

    Returns:
        entropy_df (DataFrame): dataframe of entropy values, index corresponds to movement periods and columns are names 

    """

    # initialize list of entrolies for movement periods
    entropy_measures = []

    # loop through movement periods
    for r in ruck_slices_oriented:

        # initialize this movement periods entropy Series
        ruck_entropy = pd.Series(index=names)

        # loop through soldiers
        for name in names:

            # get this soldier
            this_solider = r[[c for c in r.columns if name in c]]

            # get a histopgram of this soldiers 2D PACS locations
            hist = np.histogram2d(this_solider[this_solider.columns[0]], this_solider[this_solider.columns[1]], bins=100, range = [[-50, 50], [-50, 50]], density=True)
            
            # append this entropy value to this movement periods Serues
            ruck_entropy[name] = shannon_entropy(hist[0])
        
        # append this movement period Series to list
        entropy_measures.append(ruck_entropy)
    
    # concat entropy Series from movement periods to dataframe
    entropy_df = pd.concat(entropy_measures, axis=1).T.reset_index(drop=True)

    return entropy_df





def VAR_model(ruck_slices_oriented, names, time_window = 100, resid = False):
    """
    a vector autoregressive model for each soldier in their PACS coord system

    Args:
        ruck_slices_oriented (list): list of PACS oriented dataset dfs
        names (list): list of names
        time_window (int, optional): rolling time window for predictions. Defaults to 100.
        resid (bool, optional): using the resudials of the model as the error if True, otherwise predict a 'test'. Defaults to False.

    Returns:
        VAR_errs (list): list of dataset dfs that show each soldiers VAR error over time
    """

    # initialize list
    errs = []

    # loop throug oriented movement periods (rucks)
    for ruck in tqdm(ruck_slices_oriented):

        # intialize list for this soldier
        soldier_errs = []

        # loop through soldiers
        for name in names:

            # get this soldier's data
            this_soldier = ruck[[c for c in ruck.columns if name in c]].dropna()

            # initialise list for df
            errs_for_df = []

            # make rolling window ourselves:
            for idx in range(len(ruck)-time_window):

                # get this time window
                x = this_soldier[idx:idx+time_window]

                # split training and testing
                train, test = x[:int(len(x)*0.95)], x[int(len(x)*0.95):]
                
                # train the VAR model 
                model = VAR(train).fit()

                # get error (mean residual if resid = True)
                if resid:
                    err = abs(model.resid.mean().mean())
                # otherwise
                else:
                    forecasts = model.forecast(y=train[-10:].values, steps=len(test))
                    err = mean_squared_error(test, forecasts)

                errs_for_df.append(err)

            soldier_errs.append(pd.Series(errs_for_df, name=name))
        
        errs.append(pd.concat(soldier_errs, axis=1))

    return errs






def VARX_model(ruck_slices_oriented, names, time_window = 100, resid = False):
    """
    a vector autoregressive model with moving average and exogenous variables
    for each soldier in their PACS coord system
    exog vars are other soldiers

    Args:
        ruck_slices_oriented (list): list of oriented dataset dfs
        names (list): list of names
        time_window (int, optional): rolling time window for predictions. Defaults to 100.
        resid (bool, optional): using the resudials of the model as the error if True, otherwise predict a 'test'. Defaults to False.

    Returns:
        VARX_errs (list): list of dataframes for each movement period, showing the VARX error over time
    """

    # initialize list
    errs = []

    # loop throug oriented movement periods (rucks)
    for ruck in tqdm(ruck_slices_oriented):

        # initialize list
        soldier_errs = []

        # loop through soldiers
        for name in names:

            # get this soldier
            this_soldier = ruck[[c for c in ruck.columns if name in c]].dropna()
            
            # get other soldiers (exogenout vars)
            other_soldiers = ruck[[c for c in ruck.columns if name not in c]].dropna()

            # initialize list
            errs_for_df = []

            # make rolling window ourselves:
            for idx in range(len(ruck)-time_window):

                # get window of soldier
                x = this_soldier[idx:idx+time_window]
                # get window of others
                oth = other_soldiers[idx:idx+time_window]

                # split train and test sets
                train, test = x[:int(len(x)*0.95)], x[int(len(x)*0.95):]
                # split exogenous train and test sets
                extrain, extest = oth[:int(len(oth)*0.95)], oth[int(len(oth)*0.95):]

                # train the model
                model = VAR(endog=train, exog=extrain).fit()

                # get the error (mean residuals if resid = True)
                if resid:
                    err = abs(model.resid.mean().mean())
                else:
                    forecasts = model.forecast(y =  train[-10:].values, steps=5, exog_future=extest)
                    err = mean_squared_error(test, forecasts)

                # append err
                errs_for_df.append(err)

            # append series of errs
            soldier_errs.append(pd.Series(errs_for_df, name=name))
        
        # append time period df of errs
        errs.append(pd.concat(soldier_errs, axis=1))

    return errs



def time_series_metric_entropy(metric, range=[-10,10], bins=100):
    """
    Get an approximate entropy value from time-series metrics extracted from the GPS data
    Most metrics are in a list of dataframes, one for each movement period

    Get Approximate Entropy for each movement period and return the average

    Customize range and bin count          

    Args:
        metric (list): list of DataFrames that include a metric over time for each individual
        range (list, optional): range of bins for Entropy Histogram. Defaults to [-10,10].
        bins (int, optional): number of bins for Entropy Histogram. Defaults to 100.

    Returns:
        entropies (pd.Series): Series with the average entropy over movement periods for each individual
    """
    
    # set up entropy function for dataframes using .apply()
    def pd_entropy(column, range, bins):
        hist = np.histogram(column, range=range, bins=bins, density=True)[0]
        reslt = EH.ApEn(hist)[0][0]
        return reslt
    
    # test if series
    series_bool = isinstance(metric[0], pd.Series)
    
    # initialize entropy args
    args = (range, bins, )
   
    # initialize df (or series)
    if series_bool: entropies = pd.Series(index = np.arange(len(metric)))
    # otherwise make them names
    else: entropies = pd.DataFrame(columns = metric[0].columns)

    # loop through datasets
    for count, metric_df in enumerate(metric):

        # get the entropy df columns
        if not series_bool: 
            this_ent = metric_df.dropna().apply(pd_entropy, axis=0, args=args)
            entropies = entropies.append(this_ent, ignore_index=True)
        # if series
        else: 
            this_ent = pd_entropy(metric_df.dropna(), range, bins)
            entropies[count] = this_ent

    # return mean over movement periods 
    return entropies.mean()


