'''
Functions for loading of .CSV data for movement analysis

Datasets should be organized such that each row of the csv is one timepoint
each individual should have a column for latitude and longitude, or UTM x and y

'''

# imports
import os
import pandas as pd


def load_data(data_dir):
    """
    From a data directory, load a list of all group movement datasets

    Each file in the directory should be .csv with data from one group's movement activity
    Each row in the file represents one timepoint for one idividual and
    they should have 'latitude' and 'longitude' columns or 'UTM_x' and 'UTM_y' columns

    Args:
        data_dir (str): filepath where data is located

    Returns:
        datasets (list): list of DataFrames, one for each group movement dataset
    """

    # init datasets list
    datasets = []

    # iterate over files in data directory
    for filename in os.listdir(data_dir):

        # load and append this dataset
        file = os.path.join(data_dir, filename)

        # dropna values (making them all the same length)
        data = pd.read_csv(file).dropna()

        # name the dataframe
        data.attrs['name'] = filename.split('.')[0]

        # append the dataframe to final list
        datasets.append(data)

    return datasets



def pivot_datsets(datasets):
    """
    Pivot datasets to work with processing functions

    This pivots the dataset such that each row is one timepoint 
    all group members are included in each timepoint

    Args:
        datasets (list): list of raw Dataset dfs

    Returns:
        new_dfs (list): list of pivotted Dataset dfs
    """
    
    # init list of dfs
    new_dfs = []

    # loop through datasets
    for dataset in datasets:

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
    data_dir = os.getcwd() + '\\SampleData'