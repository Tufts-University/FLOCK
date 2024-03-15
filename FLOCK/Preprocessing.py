'''
functions for pre-processing ruck datasets
'''


from scipy.ndimage import gaussian_filter, find_objects, label
from scipy.interpolate import UnivariateSpline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta
from movingpandas import Trajectory, TrajectoryCollection, TrajectoryStopDetector
import folium
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')




def interpolate_datasets(datasets, threshold = 0.99):
    """
    Drop outlier values 
    Interpolate the missing values in the dataset

    Args:
        datasets (list of DataFrames): list of pivotted DataFrames
        threshold (float, optional): threshold for dropping ourliers as a percent. The remaining percent of first differences are dropped. Defaults to 0.99.

    Returns:
        interp_datasets (list): list of dataset dfs with outliers dropped and missing data interpolated
    """

    # initilaize list of interpolated datasets
    interp_datasets = []

    # loop through datasets to remove outliers
    # and interpolate missing points
    for df in datasets:
        # loop through columns
        for c in df:

            # only interpolate long, lat, and UTM columns
            if 'longitude' in c or 'latitude' in c or 'UTM_x' in c or 'UTM_y' in c: 
                
                # create thresholds
                max_threshold = df[c].diff().abs().quantile([threshold]).values
                
                # replace those values with NaN
                outliers_NAd = df[c].mask(~(df[c].diff().abs().values < max_threshold)).reset_index(drop=True)
                
                # # print number of outliers
                # print('Outliers: '+str(len(outliers_NAd[outliers_NAd.isna()])))
                
                # replace nan with linear interpolation
                df[c] = outliers_NAd.interpolate('linear', limit_direction='both').set_axis(df[c].index)

            else:
                # if another column (usually categorical), don't interplate
                df[c] = df[c].fillna('')

        # assert no nan values in datasets
        assert not df.isna().any().any(), 'there are NaN values in dataset'
        
        interp_datasets.append(df)

    return interp_datasets



def spline_smoothing(datasets, UTM=True, s=100):
    """
    Smooths datasets using a spline smoothing method

    Args:
        datasets (list): list of DataFrames of pivotted and interpolated datasets
        UTM (bool, optional): True if UTM data, False if raw GPS. Defaults to True.
        s (int, optional): Smoothing factor for spline method. Defaults to 100.

    Returns:
        new_datasets (list): list of smoothed DataFrames
    """
    
    # get copy of old datasets
    datasets_copy = [d.dropna().reset_index(drop=True) for d in datasets.copy()]

    # initialize list of new datasets
    new_datasets = []

    # loop through datasets
    for dataset in datasets_copy:
        
        # get names
        names = dataset.longitude.columns

        # loop thropugh names
        for name in names:

            # get UTM or raw coordinates as numpy arrays
            if UTM:
                pts = pd.concat([dataset['UTM_x', name], dataset['UTM_y', name]], axis=1, keys=['UTM_x', 'UTM_y']).to_numpy()
            else:
                pts = pd.concat([dataset['longitude', name], dataset['latitude', name]], axis=1, keys=['longitude', 'latitude']).to_numpy()
                
            # # apply a gaussian filter
            # pts = gaussian_filter(pts, sigma=0.2)

            # get distance values
            distance = np.cumsum( np.sqrt(np.sum( np.diff(pts, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]

            # make a spline for each axis
            splines = [UnivariateSpline(distance, coords, k=3, s=s) for coords in pts.T]
            points_fitted = np.vstack( spl(distance) for spl in splines ).T

            # add back to original dataframe
            if UTM:
                # organize into a dataframe (lat/long column names)
                smoothed = pd.concat([pd.Series(points_fitted.T[0], name='UTM_x'), pd.Series(points_fitted.T[1], name='UTM_y')], axis=1)
                dataset['UTM_x', name] = smoothed['UTM_x'].values
                dataset['UTM_y', name] = smoothed['UTM_y'].values
            else:
                # organize into a dataframe (lat/long column names)
                smoothed = pd.concat([pd.Series(points_fitted.T[0], name='longitude'), pd.Series(points_fitted.T[1], name='latitude')], axis=1)
                dataset['longitude', name] = smoothed['longitude'].values
                dataset['latitude' , name] = smoothed['latitude' ].values

        # append to a new list of datasets
        new_datasets.append(dataset)
            
    return new_datasets




def smooth_datasets(datasets, window=5):
    """
    for extra smoothing before velocity calculation, using a rolling average

    Args:
        datasets (list): list of smooth DataFrames, to be smoothed further 
        window (int, optional): window for rolling average. Defaults to 5.

    Returns:
        list: list of extra smooth DataFrames fro velocity calculation
    """

    smoothed_datasets = []

    for dataset in datasets.copy():
        # smooth with an average rolling window
        dataset.longitude = dataset.longitude.rolling(window, center=True).mean()
        dataset.latitude = dataset.latitude.rolling(window, center=True).mean()
        dataset.UTM_x = dataset.UTM_x.rolling(window, center=True).mean()
        dataset.UTM_y = dataset.UTM_y.rolling(window, center=True).mean()
        smoothed_datasets.append(dataset)

    return smoothed_datasets



def get_centroid(datasets, UTM=True):
    """
    Calculate the centroid for each timepoint, or for a window

    Args:
        datasets (list): list of dataset dfs
        UTM (bool, optional): True if UTM data, False if raw GPS. Defaults to True.

    Returns:
        cent_list (list): list of centroid location dataframes
    """

    # initialize list to return
    cent_list = []

    # for different units

    if UTM:
        for dataset in datasets:

            # get mean for all samples, X and Y separately
            x_mean = dataset['UTM_x'].mean(axis=1)
            y_mean = dataset['UTM_y'].mean(axis=1)

            # join x and y df for centroid coords
            centroid_coords = pd.concat([x_mean, y_mean], axis=1)
            centroid_coords.columns = ['UTM_x','UTM_y']

            # append this dataset's centroids to list
            cent_list.append(centroid_coords)
    else:
        for dataset in datasets:
            # get mean for all samples, X and Y separately
            x_mean = dataset['longitude'].mean(axis=1)
            y_mean = dataset['latitude'].mean(axis=1)

            # join x and y df for centroid coords
            centroid_coords = pd.concat([x_mean, y_mean], axis=1)
            centroid_coords.columns = ['longitude','latitude']

            # append this dataset's centroids to list
            cent_list.append(centroid_coords)



    return cent_list




def quarter_datasets(datasets, n_sections=4):
    """
    Split datasets into [n_sections] numbner of sections while retaining 'whole' as the last 'time period'

    Args:
        datasets (list): list of DataFrames to be 'quartered'
        n_sections (int, optional): number of sections to create. Defaults to 4.

    Returns:
        Qs_datasets (dict): dictionary of split datasets, keys being (whole, Q1, Q2, Q3, Q4, ...) and values being lists of dataset dfs for each time period
    """

    # initialise time period names
    part_names = []
    for i in range(n_sections): part_names.append('Q'+str(i+1))

    # initialise dict with whole dataset list
    Qs_datasets = {'whole':datasets}

    # initialize list in dict for each time period
    for p in part_names: Qs_datasets[p] = []
    
    # loop through datasets
    for data in datasets:

        # use np.split for getting quarters
        dfs = np.split(data, [(i+1)*len(data)//n_sections for i in range(n_sections-1)])

        # loop through lists and append to dict entry lists
        for df, p in zip(dfs, part_names): Qs_datasets[p].append(df)

    return Qs_datasets 



def quarter_datasets_dist(interp_datasets, n_sections=4):
    """
    Split datasets into [n_sections] numbner of sections while retaining 'whole' as the last 'time period'

    Args:
        datasets (list): list of DataFrames to be 'quartered'
        n_sections (int, optional): number of sections to create. Defaults to 4.

    Returns:
        Qs_datasets (dict): dictionary of split datasets, keys being (whole, Q1, Q2, Q3, Q4, ...) and values being lists of dataset dfs for each time period
    """

    # get centroids
    centroids = get_centroid(interp_datasets)

    # get cumulative distance over time
    # 

    def cumulative_distances(centroids):
        """
        get cumulative distance over time for splitting the dataset

        Args:
            centroids (list): list of centroid datframces from get_centroid()

        Returns:
            cumulative_distances (list): list of distances over time for each dataset
        """
        # initialize list to return
        dists = []
        # loop through centroids
        for cent in centroids: 
            # get x distances
            x_dists = cent['UTM_x'].diff()
            # get y distances
            y_dists = cent['UTM_y'].diff()
            # get euclidean distances
            euc_dists = np.sqrt(x_dists**2 + y_dists**2)
            # get cumulative distances
            cume_dists = euc_dists.cumsum()
            # append to final lsit
            dists.append(cume_dists)
        # return cumulative sum for all datasets
        return dists
    
    # get cumulative distances
    dists = cumulative_distances(centroids)

    # initialise time period names
    part_names = []
    for i in range(n_sections): part_names.append('Q'+str(i+1))

    # initialise dict with whole dataset list
    Qs_datasets = {'whole':interp_datasets}

    # initialize list in dict for each time period
    for p in part_names: Qs_datasets[p] = []
    
    # loop through datasets
    for data, dist in zip(interp_datasets, dists):

        # get quarters splits, according to distance travelled
        # get quantiles for distance values
        dist_quants = dist.quantile([x/n_sections for x in range(n_sections) if not x==0], interpolation='nearest') 
        
        # find where in series those quantile distances occur
        split_idx = dist.where(pd.concat([dist==x for x in dist_quants], axis=1).any(axis=1)).dropna().index 
        
        # get row numbers from index
        split_rows = [data.index.get_loc(x) for x in split_idx]

        # use np.split for getting quarters
        dfs = np.split(data, split_rows)

        # loop through lists and append to dict entry lists
        for df, p in zip(dfs, part_names): Qs_datasets[p].append(df)

    return Qs_datasets 





def get_slices(smoothed_datasets, datasets, UTM=True, plot=False):
    """
    Extract time-slices of movement and rest periods from the dataset
    A 'Movement' slice is when below 1m/s velocity for 5 minutes or more 
    A 'Rest' period is when above 1m/s velocity for 3 minutes or more as 'rest'

    Args:
        smoothed_datasets (list): list of extra smoothed DataFrames for velocity
        datasets (list): list of smoothed DataFrames for slicing
        UTM (bool, optional): True if UTM data, False if GPS data. Defaults to True.
        plot (bool, optional): True if plotting break times. Defaults to False.

    Returns:
        ruck_slices (list): a list of 'movement' period slices as datasets
        rest_slices (list): a list of 'rest' period slices as datasets
    """

    # init list of squad speeds
    squads_diffs = []

    # loop through datasets
    for count, dataset in enumerate(tqdm(smoothed_datasets)):

        # reset index (seconds rather than timestamp)
        if 'timestamp' not in dataset.columns:
            dataset.reset_index(inplace=True, names='timestamp')
            dataset.reset_index(inplace=True, names='seconds')
        
        # get names
        names = dataset['longitude'].columns

        # init list of soldier speeds for this dataset
        soldier_diffs = []

        # get the absolute first difference (velocity magnitude)
        for name in names: 
            # if not UTM change units
            if UTM:
                soldier_diff_x = dataset['UTM_x',name] .rolling(10).mean().diff().abs()
                soldier_diff_y = dataset['UTM_y', name].rolling(10).mean().diff().abs()
                soldier_diff = np.sqrt(soldier_diff_x**2 + soldier_diff_y**2)
            else:
                soldier_diff_x = dataset['longitude',name].rolling(10).mean().diff().abs()
                soldier_diff_y = dataset['latitude', name].rolling(10).mean().diff().abs()
                soldier_diff = np.sqrt(soldier_diff_x**2 + soldier_diff_y**2) *111139 
            soldier_diffs.append(soldier_diff)

        # concat list of soldiers first diffs, take average
        # make all nan 1.1 because of bug where some soldiers have all nan values in UTM
        soldiers_diffs = pd.concat(soldier_diffs, axis=1, keys=names).replace({np.nan:1.1})#.median(axis=1)

        # append to list of squad diffs
        squads_diffs.append(soldiers_diffs)


    print('Extracting break times')

    # time threshold for movement and rest
    # if rest:
    rest_time_threshold = 2 *60 # 2 minutes for rest
    # else:
    ruck_time_threshold = 2 *60 # 5 minutes for movement
    
    # init lists of slices
    ruck_slices = []
    rest_slices = []

    # loop through first differences of movement periods
    for squadiff, dataset in zip(squads_diffs, datasets):

        # init ruck slices list
        single_slices = []

        # get bool val for rest and ruck slices, where all soldiers below 1 m/s
        ruck_bool_val = (squadiff > 0.82).all(axis=1)
        rest_bool_val = ~ruck_bool_val

        # use find objects to find slices of consecutive True
        slices = find_objects(label(ruck_bool_val)[0])

        # loop through slices
        for slice1 in slices:
            slice1 = slice1[0]

            # get length of slice
            length = slice1.stop - slice1.start

            # if long enough slice, append to list
            if length > ruck_time_threshold:
                # extract the slice of data
                this_data_slice = dataset[slice1]
                # trim the slice to ensure mvoement or rest
                trim = 30
                trimmed_slice = this_data_slice[trim:-trim]
                # set dataframe name attr
                trimmed_slice.attrs['name'] = dataset.attrs['name']
                # append this slice
                single_slices.append(trimmed_slice)
        # append all slives
        ruck_slices.append(single_slices)
        
        # do the same as above but for rest
        single_slices = []
        slices = find_objects(label(rest_bool_val)[0])
        for slice1 in slices:
            slice1 = slice1[0]
            length = slice1.stop - slice1.start
            if length > rest_time_threshold:
                this_data_slice = dataset[slice1]
                trim = 15
                trimmed_slice = this_data_slice[trim:-trim]
                trimmed_slice.attrs['name'] = dataset.attrs['name']
                single_slices.append(trimmed_slice)
        rest_slices.append(single_slices)

    # plot the break times if plot is True
    if plot:
        
        fig, ax = plt.subplots(len(datasets),1, figsize=(10,10))

        print('Plotting break times')

        # loop through first differences of squads
        for count, (rests, rucks, dataset, sq_diff) in enumerate(zip(rest_slices, ruck_slices, datasets, squads_diffs)):

            # get rest timestamps
            rest_timestamps = pd.concat([r.index.to_series() for r in rests])
            ruck_timestamps = pd.concat([r.index.to_series() for r in rucks])

            # plot soldier times
            ax[count].plot(sq_diff) 
            max_lim = sq_diff.max().max()
            max_lim = 3
            # ax[count].fill_between(soldiers_diffs.index, 0,max_lim, where=(soldiers_diffs <= 1).all(axis=1) , color='red', alpha=0.5, transform=ax[count].get_xaxis_transform())
            ax[count].fill_between(sq_diff.index, 0,max_lim, where=pd.Series([x in rest_timestamps for x in dataset.index.to_series()]) , color='red', alpha=0.5, transform=ax[count].get_xaxis_transform())
            ax[count].fill_between(sq_diff.index, 0,max_lim, where=pd.Series([x in ruck_timestamps for x in dataset.index.to_series()]) , color='green', alpha=0.5, transform=ax[count].get_xaxis_transform())
            ax[count].set_ylim([0,max_lim])
            ax[count].set_ylabel(dataset.attrs['name'], rotation='vertical')
            ax[count].hlines(0.82, 0, 1)
            # import pdb
            # pdb.set_trace()
            ax[count].set_xticks(np.arange(len(dataset.index))[::900], [x[11:16] for x in dataset.index.to_series()[::900].values])

        fig.suptitle('Where all soldiers below 1 m/s for at least 2 minutes')
        fig.tight_layout()
        # plt.show()
        fig.savefig(os.getcwd() + '\\Figures\\RestTest.png')
        plt.close('all')


    return ruck_slices , rest_slices




def get_slices_byArea(interp_datasets, plot=False):
    """
    Get stop periods where all group memebers are stopped within 100m for at least 120s 
    (0.833m/s is the reference defined slow walking speed)
    Using MovingPandas stop detection for each individual 
    Finding overlapping stops that include the whole group

    returns movement periods, rest periods and all stop information (for finding pre/post dynamics)

    Args:
        interp_datasets (list): list of dataframes with interpolated datasets
        plot (bool, optional): True if tha map should be plotted and displayed (for jupyter notebooks). Defaults to False.

    Returns:
        move_slices (list): list of lists of movement period dataframes for each squad
        rest_slices (list): list of lists of movement period dataframes for each squad
        all_stops (list): list of lists of stop information for each squad for each stop
    """

    # initialize lists to return
    move_slices = []
    rest_slices = []
    all_stops = []

    print('Extracting movement periods')

    # loop through datasets
    for data in tqdm(interp_datasets):
        
        # change from str index to datetime index
        data.index = pd.to_datetime(data.index)

        # init list of trajs
        trajs = []

        # get names
        snames = data.longitude.columns.tolist() 
        sq_name = data.attrs['name']
        
        # make a consistent color dictionary for plotting
        color_dictionary = dict(zip(snames, sns.color_palette(as_cmap=True)[:len(snames)]))


        # loop through names
        for name in snames:

            # get this soldier
            this_soldier = data[[x for x in data.columns if name in x]] .reset_index(inplace=False, names='time')

            # get trajectory
            traj = Trajectory(this_soldier, traj_id = str(name), x=('longitude', name), y=('latitude', name), t='time')

            # append trajectory
            trajs.append(traj)

        # create a trajectory collection
        tc = TrajectoryCollection(trajs, sq_name)

        # detect stop points
        detector = TrajectoryStopDetector(tc)
        stops = detector.get_stop_points(max_diameter = 100, min_duration = timedelta(seconds=120))

        # find unique overlapping stop points that include all soldiers

        # intialize list of stops for this squad
        break_starts = []
        break_stops = []
        full_stops = []
        stop_slices_sq = []
        move_slices_sq = []

        # make pd Interval Array for finding overlaps
        stop_points_arr = pd.arrays.IntervalArray.from_arrays(stops.start_time, stops.end_time, closed=None)

        # loop through stops to compare others and find overlapping segments
        for st_1 in stop_points_arr:

            # initialise overlap test list (of bools)
            test=[]

            # loop through other stops
            for st_2 in stop_points_arr: 

                # find if overlapping
                test.append(st_1.overlaps(st_2))

            # assure that all soldiers included in stop point 
            if not all([name in stops[test].drop('geometry', axis=1).traj_id.to_string() for name in snames]):
                continue

            # assure no repeats
            repeat_test = []
            for st_time in stops[test].drop('geometry', axis=1).index.to_list():
                repeat_test.append(any([st_time in all_s.index.to_list() for all_s in full_stops]))

            # if repeated stop_time, continue to next comparison
            if any(repeat_test): continue

            # append the stop info (for finding the order of entering and exiting common stop)
            full_stops.append(stops[test])

        # assure no overlapping segments, combine if so
        new_stops = []

        # loop through stops
        for stp in full_stops:

            # get this stops start and end 
            str_time, stp_time = stp.drop('geometry', axis=1).start_time.min(), stp.drop('geometry', axis=1).end_time.max()

            # initialise overlap test
            overlap_test = []

            # loop through other stops
            for all_s in full_stops:
                test_str_time = all_s.drop('geometry', axis=1).start_time.min()
                overlap_test.append(np.logical_and(str_time < test_str_time, stp_time > test_str_time))

            # if no overlaps, append to new break list
            if not any(overlap_test):

                # append to new break list
                new_stops.append(stp)
                
            # if overlaps, combine with overlapping and append
            else: 

                # get overlapping stops
                overlapping_stops = [fs for fs, ov in zip(full_stops, overlap_test) if ov]

                # make list of overlapping stops to concatenate
                to_concat = [stp]
                for ovs in overlapping_stops: to_concat.append(ovs)

                # concat overlapping stops 
                combined = pd.concat(to_concat)

                # append new stop
                new_stops.append(combined)

                # pop out overlapping stops to not test for overlap
                for i in np.where(np.array(overlap_test))[0]: _ = full_stops.pop(i)
                    
                
        for nstop in new_stops:

            # get start and end timepoints for this break
            break_start = nstop.drop('geometry', axis=1).start_time.min().to_pydatetime().replace(tzinfo = data.index.tzinfo)
            break_end = nstop.drop('geometry', axis=1).end_time.max().to_pydatetime().replace(tzinfo = data.index.tzinfo)
            
            # append timings to break times (break beginning and end)
            break_starts.append(break_start)
            break_stops.append(break_end)

            # seconds to slice off of rest periods
            rest_buffer_time_s = 30
            break_start_buff = break_start+timedelta(seconds=rest_buffer_time_s)
            break_end_buff = break_end-timedelta(seconds=rest_buffer_time_s)

            # append slices of stop times to stop_slices_sw list
            stop_slices_sq.append(data[break_start_buff:break_end_buff])
        
        
        # insert beginning of recording for initial 'break_stop'
        # append end of recording for final 'break_start'
        break_stops.insert(0, data.index.min().to_pydatetime())
        break_starts.append(data.index.max().to_pydatetime())

        # how many secodns to shave off at the beginning and end of movement periods
        movement_buffer_time_s = 60

        # loop through break stops and starts
        for time_1, time_2 in zip(break_stops, break_starts):
            time_1_buff = time_1+timedelta(seconds=movement_buffer_time_s)
            time_2_buff = time_2-timedelta(seconds=movement_buffer_time_s)
            # be sure df is not empty after buffer
            if not data[time_1_buff:time_2_buff].empty:
                move_slices_sq.append(data[time_1_buff:time_2_buff])
    
        # append to final lists
        move_slices.append(move_slices_sq)
        rest_slices.append(stop_slices_sq)
        all_stops.append(new_stops)

        if plot:
            
            # create folium map
            m = folium.Map(
                zoom_start=13.5,
                tiles='OpenStreetMap',
                width=1024,
                height=768,
                control_scale = True
            )

            # path_feature_group = folium.FeatureGroup(name='Path')

            for name in snames:
                # get this soldier
                this_soldier = data[[x for x in data.columns if name in x]] 
                folium.PolyLine(this_soldier[['latitude','longitude']], color=color_dictionary[name], smooth_factor=0, tooltip=name + " path").add_to(m)
            
            # rest_feature_group = folium.FeatureGroup(name='Breaks')
            for f in new_stops:
                pts_x = np.mean([g for g in f.geometry.x])
                pts_y = np.mean([g for g in f.geometry.y])
                folium.Circle(radius=50, location=[pts_y, pts_x] ,color='crimson', fill=True, tooltip='Rest '+ str(f.end_time.max() - f.start_time.min()), popup='Rest '+ str(f.end_time.max() - f.start_time.min())).add_to(m)
            
            display(m)
            


    return move_slices, rest_slices, all_stops




# def get_break_times(datasets, centroids, rest=False, UTM=False):
#     '''
#     find break times. Used for segmenting datasets into periods of movement

#     Use centroid .diff() to find timepoints 

#     criteria: 
#     Velocity below __ rate (average diff())
#     No breaks for __ minutes prior
#     No Break for __ minutes later
#     Break lasts for __ minutes

#     Maybe using forward angle (theta) information

#     if rest=True, we are extracting the rest periods, otherwise we are extracting movement periods
#     '''

#     break_times = []

#     fig, ax = plt.subplots(len(datasets),1, figsize=(5,10))

#     print('Plotting break times')

#     centroid_diffs = []

#     for count, (centroid, dataset) in enumerate(zip(tqdm(centroids), datasets)):
#         # reset index (seconds rather than timestamp)
#         centroid.reset_index(drop=True, inplace=True)
#         # # change units to meters rather than coordinates (rough)
#         # centroid['longitude'] = centroid['longitude'] * 111139  * math.cos(centroid['latitude'].mean())
#         # centroid['latitude']  = centroid['latitude']  * 111139
#         # smooth the centroid
#         smooth = 30
#         centroid = centroid.rolling(smooth, center=True).mean()
#         if UTM:
#             centroid_diff = centroid.diff().abs().sum(axis=1)
#         else:
#             centroid_diff = centroid.diff().abs().sum(axis=1) *111139        
#         centroid_diffs.append(centroid_diff)
#         ax[count].plot(centroid_diff) 
#         max_lim = centroid_diff.max().max()
#         ax[count].fill_between(centroid.index, 0,max_lim, where=centroid_diff < 1, color='red', alpha=0.5, transform=ax[count].get_xaxis_transform())
#         ax[count].set_ylim([0,max_lim])
#         ax[count].set_ylabel(dataset.attrs['name'], rotation='vertical')
#         ax[count].set_xticks(centroid.index.values[::1000], centroid.index.values[::1000]//60)

#     fig.suptitle('Centroid diff (averaged over 30 seconds)')
#     fig.tight_layout()
#     fig.savefig(os.getcwd() + '\\Figures\\RestTest.png')
#     plt.close('all')

#     print('Extracting break times')

#     # time threshold for movement and rest
#     if rest:
#         time_threshold = 3 *60 # 3 minutes for rest
#     else:
#         time_threshold = 5 *60 # 5 minutes for movement
    
#     ruck_slices = []
#     all_cents = []

#     for centdiff, dataset, centroid in zip(centroid_diffs, datasets, centroids):
#         # reset index to be seconds from start, saving timestamps
#         dataset.reset_index(inplace=True, names='timestamp')
#         dataset.reset_index(inplace=True, names='seconds')
#         single_slices = []
#         single_cent = []
#         # Rough conversion to meters from coordinates 
#         if rest:
#             bool_val = centdiff < 1
#         else:
#             bool_val = centdiff > 1

#         slices = find_objects(label(bool_val)[0])
#         for slice1 in slices:
#             slice1 = slice1[0]
#             length = slice1.stop - slice1.start
#             if length > time_threshold:
#                 this_data_slice = dataset[slice1]
#                 cent_addit = 100
#                 this_cent_slice = centroid[slice(slice1.start, slice1.stop + cent_addit)]
#                 trim = 30
#                 trimmed_slice = this_data_slice[trim:-trim]
#                 trimmed_slice.attrs['name'] = dataset.attrs['name']
#                 # trimmed_cent = this_cent_slice[trim:-trim]
#                 single_slices.append(trimmed_slice)
#                 # single_cent.append(trimmed_cent)
#                 single_cent.append(this_cent_slice)
#         ruck_slices.append(single_slices)
#         all_cents.append(single_cent)

#     return ruck_slices, all_cents







if __name__ == '__main__':
    '''
    Use this for testing and maybe for putting together preprocessing pipeline
    '''
    print(None)