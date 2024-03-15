'''
Functions for extracting features from the Path Adapted Coordinate System (PACS) data

'''

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, wasserstein_distance
from tqdm import tqdm


def get_SEIs(ruck_slices_oriented, names):
    """
    get the spatial explioration index for each soldier during each movement period from oriented data
    SEI is considered the distance of a soldier from their own average location during a movement period
    we include distances over the whole period, so that mean, max ect can be captured elsewhere

    Args:
        ruck_slices_oriented (list): list of oriented dataset dfs
        names (list): list of names

    Returns:
        SEIs (list): list of dataframes with the individuals SEI (distance to their mean location) over time for each movement period
    """

    # initialize list
    SEIs = []

    # loop throug oriented movement periods (rucks)
    for ruck in tqdm(ruck_slices_oriented):
        
        # initialize list
        SEI_for_df = []

        # loop thorugh names
        for name in names:

            # get this individual
            this_soldier = ruck[[c for c in ruck.columns if name in c]]
            
            # get average soldier positoin
            this_soldier_mean = this_soldier.mean()
            
            # get soldier distances to average position
            this_soldier_dists = this_soldier - this_soldier_mean
            
            # euclidean distance 
            this_soldier_dist = np.sqrt(this_soldier_dists[name+' longitude']**2+this_soldier_dists[name+' latitude']**2)

            # append to list 
            SEI_for_df.append(pd.Series(this_soldier_dist, name=name))

        # append list as df to final list 
        SEIs.append(pd.concat(SEI_for_df, axis=1))

    return SEIs




def get_neighbor_dists(move_slices_oriented, names):
    """
    Get distance to nearest neightbor
    both left/right and front/back (X and Y)

    Args:
        move_slices_oriented (list): list of oriented dataset dfs
        names (list): list of names

    Returns:
        x_neighbors (list): list of dfs with nearest neighbor distances in the X direction
        y_neighbors (list): list of dfs with nearest neighbor distances in the Y direction
    """

    # initilaize list
    x_neighbors = []
    y_neighbors = []

    # loop through datasets
    for df in move_slices_oriented:

        # initialize dfs
        x_neighbors_df = pd.DataFrame()
        y_neighbors_df = pd.DataFrame()

        # get unique names and drop 'unnamed'
        names = [x for x in list(set(names)) if 'Unnamed:' not in x]
                
        # get individual soldier data
        for ID in names:

            # separate to this soldier and other soldiers
            indiv_data = df[[c for c in df.columns if ID in c]]
            other_data = df[[c for c in df.columns if ID not in c]]

            # separate x and y
            indiv_x = indiv_data[ID + ' longitude']
            indiv_y = indiv_data[ID + ' latitude']
            other_x = other_data[[c for c in other_data.columns if 'longitude' in c]]
            other_y = other_data[[c for c in other_data.columns if 'latitude' in c]]

            # get distance to all neighbors
            x_diff = other_x.sub(indiv_x, axis=0).abs()
            y_diff = other_y.sub(indiv_y, axis=0).abs()

            # get closest distance
            max_x = x_diff.min(axis=1)
            max_y = y_diff.min(axis=1)

            # append to df
            x_neighbors_df = pd.concat([x_neighbors_df, max_x], axis=1)
            y_neighbors_df = pd.concat([y_neighbors_df, max_y], axis=1)
        
        # add collumn names (individual's names)
        x_neighbors_df.columns = names
        y_neighbors_df.columns = names

        # add name attr to df
        x_neighbors_df.attrs['name'] = y_neighbors_df.attrs['name'] = df.attrs['name']
        
        # append to final list
        y_neighbors.append(y_neighbors_df)
        x_neighbors.append(x_neighbors_df)

    return x_neighbors, y_neighbors




def LW_ratio(ruck_slices_oriented):
    """
    get the length to width ratio
    furthest separation front/back (length) divided by furthest separation side/side (width)

    Args:
        ruck_slices_oriented (list): list of oriented dataset dfs

    Returns:
        LW_ratios (list): list of L/W ratios over time for each movement period
    """

    # initialise list
    LW_ratios = []

    # Loop through movement periods
    for ruck in ruck_slices_oriented:

        # get range of X and Y
        Xmin = ruck[[c for c in ruck.columns if 'longitude' in c]].min(axis=1)
        Xmax = ruck[[c for c in ruck.columns if 'longitude' in c]].max(axis=1)
        Ymin = ruck[[c for c in ruck.columns if 'latitude' in c]].min(axis=1)
        Ymax = ruck[[c for c in ruck.columns if 'latitude' in c]].max(axis=1)

        # get peak to peak (max difference) (length and width)
        Widths = Xmax - Xmin
        Lengths = Ymax - Ymin

        # get ratio
        LW_ratios.append(Lengths/Widths)

    return LW_ratios




def dist_consistency_Ftest(ruck_slices_oriented, names):
    """
    Measure the consistency of soldier positions across movement periods
    F-test statistic as a metric for each soldier's oriented position consistency across movement periods

    Args:
        ruck_slices_oriented (list): list of oriuented datasets as dfs
        names (list): list of soldier names

    Returns:
        X_ftest (Series): A series of F-statistic values for each soldier in the X axis
        Y_ftest (Series): A series of F-statistic values for each soldier in the Y axis
    """
    
    # initialize f test lists
    X_ftest = pd.Series(index=names)
    Y_ftest = pd.Series(index=names)

    # loop through names
    for name in names:

        # if there is not more than one movement period, return NaN
        if len(ruck_slices_oriented)>1:

            # initialize lists 
            this_soldier_X = []
            this_soldier_Y = []

            # loop thorough movement periods
            for ruck in ruck_slices_oriented:

                # get this soldier
                this_soldiers_ruck = ruck[[c for c in ruck if name in c]]

                # get X and Y distributions for each movement period
                # this_soldier_X.append(this_soldiers_ruck[[c for c in this_soldiers_ruck if 'longitude' in c]])
                # this_soldier_Y.append(this_soldiers_ruck[[c for c in this_soldiers_ruck if 'latitude'  in c]])
                this_soldier_X.append(np.histogram(this_soldiers_ruck[[c for c in this_soldiers_ruck if 'longitude' in c]], bins=100, range=[-50,50], density=True)[0])
                this_soldier_Y.append(np.histogram(this_soldiers_ruck[[c for c in this_soldiers_ruck if 'latitude'  in c]], bins=100, range=[-50,50], density=True)[0])
            
            # get f-statistic ove movement periods
            X_ftest[name] = f_oneway(*this_soldier_X).statistic
            Y_ftest[name] = f_oneway(*this_soldier_Y).statistic

        # nan if only1 movement period
        else:
            X_ftest[name] = np.nan
            Y_ftest[name] = np.nan


    return X_ftest, Y_ftest




def dist_consistency_wasserstein(ruck_slices_oriented, names):
    """
    get the wasserstein distance, a metric for each soldier's PACS location consistency across movement periods

    Args:
        ruck_slices_oriented (list): oriented soldier positions for movement periods
        names (list): list of names.

    Returns:
        X_wass_df: (Series): average Wasserstein distance for each soldier in the X axis
        Y_wass_df: (Series): average Wasserstein distance for each soldier in the Y axis
    """
    
    # initialize wasserstein distance dfs
    X_wass_df = pd.DataFrame(columns=names)
    Y_wass_df = pd.DataFrame(columns=names)

    
    # if there is not more than one movement period, return NaN
    if len(ruck_slices_oriented)==1:
        X_wass = np.nan
        Y_wass = np.nan
    else:

        # get pairs of ruck events:
        ruck_pairs = [(a, b) for idx, a in enumerate(ruck_slices_oriented) for b in ruck_slices_oriented[idx + 1:]]

        # loop through movement period pairs
        for ruck_p in ruck_pairs:
    
            # initialize wasserstein distance for this pair
            X_wass = pd.Series(index=names)
            Y_wass = pd.Series(index=names)

            # loop through names
            for name in names:

                # get this soldier
                this_soldiers_ruck_1 = ruck_p[0][[c for c in ruck_p[0] if name in c]]
                this_soldiers_ruck_2 = ruck_p[1][[c for c in ruck_p[1] if name in c]]

                # get X and Y distributions for each movement period
                this_soldier_X1 = this_soldiers_ruck_1[[c for c in this_soldiers_ruck_1 if 'longitude' in c]]
                this_soldier_Y1 = this_soldiers_ruck_1[[c for c in this_soldiers_ruck_1 if 'latitude'  in c]]
                this_soldier_X2 = this_soldiers_ruck_2[[c for c in this_soldiers_ruck_2 if 'longitude' in c]]
                this_soldier_Y2 = this_soldiers_ruck_2[[c for c in this_soldiers_ruck_2 if 'latitude'  in c]]
            
                # get wasserstein distance from movement periods
                X_wass[name] = wasserstein_distance(np.histogram(this_soldier_X1, range=(-50,50), bins=100)[0] , np.histogram(this_soldier_X2, range=(-50,50), bins=100, density=True)[0] )
                Y_wass[name] = wasserstein_distance(np.histogram(this_soldier_Y1, range=(-50,50), bins=100)[0] , np.histogram(this_soldier_Y2, range=(-50,50), bins=100, density=True)[0] )

            X_wass_df = X_wass_df.append(X_wass, ignore_index=True)
            Y_wass_df = Y_wass_df.append(Y_wass, ignore_index=True)


    return X_wass_df.mean(), Y_wass_df.mean()





if __name__ == '__main__':
    '''
    Use this for testing
    '''
    print(None)