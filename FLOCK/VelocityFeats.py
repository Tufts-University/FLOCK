"""
Functions for calculation of velocity 
and functions for feature extraction related to that

Including velocity over time, velocity differences over time within the group (max-min and varaince)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm



def get_velocities(smoothed_movements, names, UTM=True):
    """
    Get velocities from smoothed datasets using the first difference

    Args:
        smoothed_movements (list): list of extra smooth DataFrames
        names (list): list of individual's names
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.

    Returns:
        vel_dfs (list): DataFrames of first differences from the 
    """

    # initialise list of velocity dfs
    vel_dfs = []
    
    # Choose units 
    if UTM:
        X = 'UTM_x'
        Y = 'UTM_y'
    else:
        X = 'longitude'
        Y = 'latitude'
    

    # loop through oriented movement periods (movements)
    for movement in tqdm(smoothed_movements):
        
        # init list of this dataframe's velocities
        vel_for_df = []

        # loop through soldiers
        for name in names:

            # get this soldier
            this_soldier = pd.concat([movement[X,name], movement[Y,name]], axis=1)

            # get soldier velocity .diff()
            this_soldier_vels = this_soldier.diff()

            # euclidean
            this_soldier_vel = np.sqrt(this_soldier_vels[X]**2 + this_soldier_vels[Y]**2)

            # append this soldier
            vel_for_df.append(this_soldier_vel)
        
        # create df from list of soldier velocities, append to final list
        vel_dfs.append(pd.concat(vel_for_df, axis=1))

    return vel_dfs




def get_accel(vel_dfs, names, UTM=True):
    """
    Get acceleration from velocity datasets using another differencing

    Args:
        vel_dfs (list): list of velocity (first difference) DataFrames
        names (list): list of solider names
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.

    Returns:
        acc_dfs (list): DataFrames containing accelleration data
    """

    assert not names==None, 'Input names to get_velocities'

    acc_dfs = []
    
    # Choose units 
    if UTM:
        X = 'UTM_x'
        Y = 'UTM_y'
    else:
        X = 'longitude'
        Y = 'latitude'
    
    # loop through oriented movement periods (movements)
    for movement_vel in tqdm(vel_dfs):
        
        # initialize list of acc dfs for thsi movement period
        acc_for_df = []

        # loop thorugh soldiers
        for name in names:

            # this_soldier = pd.concat([movement[X,name], movement[Y,name]], axis=1)
            this_soldier = movement_vel[name]

            # get soldier velocity .diff()
            this_soldier_vels = this_soldier.diff()

            # # euclidean
            # this_soldier_vel = np.sqrt(this_soldier_vels[X]**2 + this_soldier_vels[Y]**2)

            # append this soldier
            acc_for_df.append(this_soldier_vels)
        
        # create df and append to final list
        acc_dfs.append(pd.concat(acc_for_df, axis=1))

    return acc_dfs


def get_vel_feats(vel_dfs):
    """
    Get some velocity features from velocity datasets
    Variance: how each soldiers velocity varies throughout each movement period
    Difference: how different is the speed of the fastest soldier vs speed of slowest soldier

    Args:
        vel_dfs (list): list of velocity (first difference) DataFrames

    Returns:
        vel_var (list): list of varainces for the velocity dataframe
        vel_diff (list): list of max difference over time for soldier velocities
    """

    # initialize lists
    vel_var, vel_diff = [], []

    # loop through movement periods 
    for vel_df in vel_dfs:

        # get varaince of this movement period per soldier
        vel_var.append(vel_df.var(axis=1))
        vel_diff.append(vel_df.max(axis=1) - vel_df.min(axis=1))

    return vel_var, vel_diff



def acc_corr(acc_dfs, names=None, time_window=30):
    '''
    get the correlation of acceleration across soldiers
    
    input: list of acceleration dfs for movement periods
    
    output: avearage over correlation matrix for each timepoint
    
    '''
    assert not names==None, 'Input names to acc_corr'
    
    corr_dfs = []
    
    # loop through movement periods
    for acc_df in tqdm(acc_dfs):
        # init list
        corr_windows = []
        # loop through time windows (rolling, stride=1)
        for idx in range(len(acc_df)-time_window):
            # get data window
            data_window = acc_df[idx:idx+time_window]
            # get correlations in window, take average of matrix
            corr_windows.append(data_window.corr().mean())
        
        corr_dfs.append(pd.concat(corr_windows, axis=1).T)

    
    return corr_dfs




def vel_corr(vel_dfs, time_window=30):
    """
    get the correlation of velocity across soldiers

    Args:
        vel_dfs (list of DataFrames): list of velosity DataFrames
        time_window (int, optional): time window for calculating at velocity correlation. Defaults to 30.

    Returns:
        corr_dfs (list of DataFrames): list of dataframes with velocity correlation within the group over time
    """
    
    # initialize list
    corr_dfs = []
    
    # loop through movement periods
    for vel_df in tqdm(vel_dfs):

        # init list
        corr_windows = []

        # loop through time windows (rolling, stride=1)
        for idx in range(len(vel_df)-time_window):

            # get data window
            data_window = vel_df[idx:idx+time_window]

            # get correlations in window, take average of matrix
            corr_windows.append(data_window.corr().mean())
        
        # append to final list
        corr_dfs.append(pd.concat(corr_windows, axis=1).T)

    
    return corr_dfs






if __name__ == '__main__':
    '''
    Use this for testing and maybe for putting together preprocessing pipeline
    '''
    print(None)

