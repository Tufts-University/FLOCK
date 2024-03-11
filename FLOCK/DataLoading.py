'''
Functions for loading of GPX data for Ruck analysis

Options for loading UTM data instead as well

'''

# imports
import os
import gpxpy
import pandas as pd


def load_data(data_dir):
    """
    From a data directory, load a list of all datasets

    Args:
        data_dir (str): filepath where data is located

    Returns:
        datasets (list): list of DataFrames, one for each dataset
    """

    # init datasets list
    datasets = []

    # iterate over files in data directory
    for filename in os.listdir(data_dir):

        # skip if any of these in filename
        if 'all' in filename: continue
        if 'RuckInfil' in filename: continue
        if 'UTM' in filename: continue

        # load and append this dataset
        file = os.path.join(data_dir, filename)
        data = pd.read_csv(file).dropna()
        data.attrs['name'] = filename#.split('_')[2]
        datasets.append(data)

    return datasets



def pivot_datsets(datasets):
    """
    Pivot datasets to work with processing functions

    Args:
        datasets (list): list of raw Dataset dfs

    Returns:
        new_dfs (list): list of pivotted Dataset dfs
    """
    
    # init list of dfs
    new_dfs = []

    # loop through datasets
    for dataset in datasets:

        # for some reason, one of the members in one squad was duplicated (in only UTM data)
        if dataset.duplicated().any():
            dataset = dataset[~dataset.duplicated()]
        
        # pivot dataset
        new_df = pd.pivot(dataset, columns='Member_ID', index='time')

        # add name attr to df
        new_df.attrs['name'] = dataset.attrs['name']
        
        # append df to return list 
        new_dfs.append(new_df)

    return new_dfs



if __name__ == '__main__':
    '''
    Executed when calling this module alone [SampleDataLoading()]

    Usually for a demo of the functions in the module as returns not available
    '''
    data_dir = os.getcwd() + '\\Data\\csv'
    data = load_data(data_dir)
    import pdb
    pdb.set_trace()