'''
Path Adapted Coordinate System

transformation and feature extraction

'''


import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import random
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.integrate import quad



# Helper functions for PACS transformation

# Rotation function
def rotate(x, y, radians):
    """
    rotates a point around the origin counterclockwise

    Args:
        x (float): original coordinate 
        y (float): original coordinate 
        radians (_type_): radians to be rotated

    Returns:
        tuple (float, float): rotated X and Y coordinates
    """

    # (-radians) for clockwise and add +90 degrees (+y direction)
    radians = -radians+(math.pi/2)
    
    # get new x coord
    xx = x * math.cos(radians) - y * math.sin(radians)
    
    # get new y coord
    yy = x * math.sin(radians) + y * math.cos(radians)
    return xx, yy


# Distance to spline function
def dp2s(t,sx,sy,x0,y0):
    """
    Define the distance from point to spline

    Args:
        t (float): timepoint (normalized to 1 for spline)
        sx (scipy.UnivariateSpline object): spline of x
        sy (scipy.UnivariateSpline object): spline of y
        x0 (float): x coordinate
        y0 (float): y coordinate

    Returns:
        d (float): distance from point
        g (float): gradient at point on spline
    """

    # Spline to point distances
    tmpx = sx(t)-x0
    tmpy = sy(t)-y0
    
    # The Euclidean distance
    d = math.sqrt(tmpx**2 + tmpy**2)
    
    # The gradient
    g = 2*tmpx*sx.derivative(n=1)(t) + 2*tmpy*sy.derivative(n=1)(t)

    return d, g


# Force convergence function
def force_converge(t0, args4d):
    """
    Force the minimization function used to converge on the nearest spline point
    iterate 100x before incrreasing (doubling) the tolerance

    Args:
        t0 (int): initial guess
        args4d (list): packaged parameters for the distance calculation [x spline, y spline, x coord, y coord]

    Returns:
        rac (scipy.optimize._optimize.OptimizeResult object) : result of converged minimization problem
    """

    # Compute the minimzation which will result in new new cordinate:
    rac = minimize(dp2s, t0, args=args4d, jac = True, 
                method='BFGS', tol = 1e-6, options={'disp' : False})
    
    # initialize iteration count
    i = 0
    
    # initialize tolerance
    tol = 1e-6
    
    # loop until convergence
    while not rac.success:
        
        # make new guess (with random)
        t0_rand = t0 + (random.random()*10)
        
        # compute minimization again
        rac = minimize(dp2s, t0_rand, args=args4d, jac = True, 
                    method='BFGS', tol = tol, options={'disp' : False})
        
        # add one to iteration count
        i += 1
        
        # increase colorance if not converging in 100 iterations
        if i>100:
            
            # double tolerance level
            tol = tol*2
            
            # reset iteration count
            i=0
            pass
    
    # return converged
    return rac



def PACS_transform(datasets, UTM=True):
    """
    Transforming group location data into a path-adapted coordinate system

    Using the spline-smoothed trajectory path of the group, 
    we get the distance of the individual to the group's spline path (left or right) as the X coordinate
    we then find the distance along the spline path from the location of the groups center to the individual's location as the new Y coordinate

    Developed with Dr. Eric Miller for the CABCS at Tufts University

    Args:
        datasets (list): list of dataset dfs to re-orient
        UTM (bool, optional): True if UTM data, False if raw GPS. Defaults to True.

    Returns:
        oriented_datasets (list): list of PACS oriented datasets in dfs
    """

    # additional helper function
    # Integration function
    def integrand(x):
        '''Finding distance along spline of two points'''
        dx_dt = splines[0].derivative()(x)
        dy_dt = splines[1].derivative()(x)
        return np.sqrt(dx_dt**2 + dy_dt**2)
    
    print('re-orienting data')

    # initalize list to return
    oriented_datasets = []

    # loop thorugh datasets
    for im_count, dataset in enumerate(tqdm(datasets)):

        # # drop all nan columns
        dataset = dataset.dropna()
        if dataset.empty:
            print('all nans')
            continue

        # choose units (UTM or coords)
        if UTM:
            pts = pd.concat([dataset['UTM_x'].mean(axis=1), dataset['UTM_y'].mean(axis=1)], axis=1, keys=['longitude', 'latitude']).to_numpy()
        else:
            pts = pd.concat([dataset['longitude'].mean(axis=1), dataset['latitude'].mean(axis=1)], axis=1, keys=['longitude', 'latitude']).to_numpy()
            
        # get distance values
        distance = np.cumsum( np.sqrt(np.sum( np.diff(pts, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)#/distance[-1]

        # different smoothing factors depending on coordinate system
        if UTM:
            s = 100
        else:
            s = 1e-8

        # make a spline for each axis
        splines = [UnivariateSpline(distance, coords, k=3, s=s) for coords in pts.T]
        points_fitted = np.vstack( spl(distance) for spl in splines ).T

        # organize into a dataframe (lat/long column names)
        smoothed_centroid = pd.concat([pd.Series(points_fitted.T[0], name='longitude'), pd.Series(points_fitted.T[1], name='latitude')], axis=1)
                
        # initialize some lists
        soldiers_datapoints = []

        # make the initial guess 0.2, then make the initial gues the previous prediction in following loops
        last_s = [0.2]*len(dataset['longitude'].columns)

        # loop through samples
        time_window=1 # for pd compatability
        for idx in tqdm(range(len(dataset.dropna())-time_window)):

            # initalize list to record this loop's prediciton, will become next loops initial guess
            this_s = []

            # get the start index
            start_idx = idx

            # get centroid window and data window
            this_window_cent = smoothed_centroid[start_idx:start_idx+time_window]
            data_window = dataset.dropna()[start_idx:start_idx+time_window]

            # initiate list for oriented soldier data
            oriented_soldiers = []

            # loop through soldiers
            for count, name in enumerate(dataset['longitude'].columns):

                # get one soldiers's data
                if UTM:
                    this_soldier = pd.concat([data_window['UTM_x',name], data_window['UTM_y',name]], axis=1)
                else:
                    this_soldier = pd.concat([data_window['longitude',name], data_window['latitude',name]], axis=1)

                # get spline locations for soldier, then get location for the centroid
                # soldier's new X coord is distance from spline (right/left)
                # soldier's new Y coord is distance from centroid spline location

                # get spline location for soldier

                # Initial guess
                t0 = last_s[count]

                # Package up parameters for the distance calculation
                args4d = (splines[0], splines[1], this_soldier.iloc[0][0], this_soldier.iloc[0][1])
                
                # Compute the minimzation which will result in new new cordinate:
                rac = force_converge(t0, args4d)
                
                # append this spline location to be the initial guess for next time point
                this_s.append(rac.x[0])

                # save this soldier's new coordinates (point on the spline, diostance from spline)
                soldier_spl_point = (rac.x[0], rac.fun)

                # find if d should be (+ or -)
                # doiny by finding current heading direction, 
                # then finding if the individual' new x coord is to the left or right of that
                # if new x coord is left, make the new x coord negative

                # soldier spline location
                test_spl_point = [spl(rac.x[0]) for spl in splines]
                
                # Get previous spl point for rotation
                previous_spl_point = [spl(rac.x[0]-1) for spl in splines]

                # get forward vector (this spl point to prevoius spl point)
                spl_forward_vector = [test_spl_point[0] - previous_spl_point[0], test_spl_point[1] - previous_spl_point[1]]

                # get angle from origin (+x) to 'forward' vector
                # Points are given to np.arctan2 as (y,x)
                forward_angle = pd.Series(np.arctan2(spl_forward_vector[1], spl_forward_vector[0]))

                # rotate the soldier point (normalized to the previous spline location)
                soldier_point = [this_soldier.iloc[0][0] - previous_spl_point[0], this_soldier.iloc[0][1] - previous_spl_point[1]]
                test_point = rotate(soldier_point[0], soldier_point[1], forward_angle)

                # if test point is negative, the point is to the left of the path, and should have a -x value
                if test_point[0] < 0:
                    # multiply d (new x) by -1
                    new_x = -rac.fun
                else:
                    # keep same d (new x)
                    new_x = rac.fun
                
                # get this centroid location on spline to find Y value for new point

                # Set initial guess
                t0 = last_s[count]

                # Package up parameters for the distance calculation
                args4d = (splines[0], splines[1], this_window_cent.iloc[0][0], this_window_cent.iloc[0][1])

                # Compute the minimzation which will result in new new cordinate:
                rac = force_converge(t0, args4d)
                
                # centroid spline point
                cent_point = (rac.fun, rac.x[0])

                # Get the (along spline) distance from centroid to this point
                start_x = cent_point[1]
                end_x = soldier_spl_point[0]

                # Integrate the integrand over the interval between the start and end points
                spl_len, _ = quad(integrand, start_x, end_x, limit=1000, epsabs=1e-5, epsrel=1e-5)
                
                # add additional Y value (len along spline c=fron centroid spline loc to coldier spline loc)
                # new_point = [new_point[0], new_point[1] + additional_y]
                new_point = [new_x, spl_len]

                # make df for appending
                new_point_df = pd.Series(new_point, index=[name +' longitude',name + ' latitude'])

                # Create list and then concat, more effecient
                oriented_soldiers.append(new_point_df)
            
            # save these points as initial guess for next iteration
            last_s = this_s

            # create df for this timepoint of all soldiers
            timepoint_oriented = pd.concat(oriented_soldiers)
            
            # append this timepoint to list of timepoints
            soldiers_datapoints.append(timepoint_oriented)
        
        # Create oriented data df
        oriented_data = pd.concat(soldiers_datapoints, axis=1).T
        oriented_data.attrs['name'] = dataset.attrs['name']

        # append this dataset to list of datasets
        oriented_datasets.append(oriented_data)

    return oriented_datasets





if __name__ == '__main__':
    '''testing'''
