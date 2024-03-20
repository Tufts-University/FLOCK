'''
To run the full analysis with all metrics

'''


# import custom functions
import os

from FLOCK import DataLoading, Preprocessing, PACS, PacsFeats, VelocityFeats, SpatialFeats, DirectionalCorrelation, ClusteringFeats, RegularityFeats
import pandas as pd
import seaborn as sns
import numpy as np

'''Loading data'''
# Initialize path to data (UTM-converted datasets)
data_dir = os.getcwd() + '\\SampleData'
# Load datasets
raw_datasets = DataLoading.load_data(data_dir)

'''Preprocessing'''
# Re-shape datasets
datasets = DataLoading.pivot_datsets(raw_datasets)
# drop outliers and interpolate datasets
interp_datasets = Preprocessing.interpolate_datasets(datasets)

'''Initialize features'''
# get names of group (squad) members
all_individuals = [x for y in [data.longitude.columns.to_list() for data in datasets] for x in y]
# get sq_names
all_squads = [s.attrs['name'] for s in datasets]
# initialize individual-level features
Indiv_feats = pd.DataFrame(index=all_individuals)
# initialize squad-level features
Squad_feats = pd.DataFrame(index=all_squads)


'''Optional Clustering step
# cluster here before extracting further features from significant cluster separately
# for more robust results where groups split up
# We assume that the defined groups do not full separate
'''

'''Optional Sectioning step
# use this to get features from different sections of the movement, to see how features change over time
# split the datasets to equal sections based on distance covered
Qs_datasets = Preprocessing.quarter_datasets_dist(interp_datasets, n_sections=8)
# loop through sets of data sections
for time_period, Qdatasets in Qs_datasets.items(): 
'''

# Otherwise just use full dataset
Qdatasets = interp_datasets
time_period = 'whole'


'''Feature Extraction'''
# Run feature extraction on the full dataset and then the sectioned datasets

# Get centroids
centroids = Preprocessing.get_centroid(Qdatasets, UTM=True)

# get slices for movement periods and break times   
move_slices, rest_slices, all_stops = Preprocessing.get_slices_byArea(Qdatasets)    

# loop through each squad and extraftr featuires for each set of 'movement periods'
for move_slice, rest, full_cent, stops in zip(move_slices, rest_slices, centroids, all_stops):

    # if move_slice for time_period is empty
    if not move_slice:
        continue

    # should always be at least 2 movement periods, if not, we split the data in half
    if not len(move_slice) > 1:
        move_slice = [move_slice[0][:len(move_slice[0])//2], move_slice[0][(len(move_slice[0])//2):]]
    
    # get colors to use throughout plotting
    names = move_slice[0].longitude.columns.tolist()

    # make a consistent color dictionary for plotting
    color_dictionary = dict(zip(names, sns.color_palette(as_cmap=True)[:len(names)]))

    # get squad name for this iteration
    sq_name = move_slice[0].attrs['name']

    # # who arrives first
    # stop_arr_order_df = pd.DataFrame(columns=stops[0].traj_id.unique())
    # for stp in stops:
    #     order = {}
    #     for name, place in zip(stp.sort_values('start_time').traj_id.unique(), np.arange(len(stp.sort_values('start_time').traj_id.unique()))): order[name] = place
    #     stop_arr_order_df = stop_arr_order_df.append(order, ignore_index=True)
    # # who leaves first
    # stop_dept_order_df = pd.DataFrame(columns=stops[0].traj_id.unique())
    # for stp in stops:
    #     order = {}
    #     for name, place in zip(stp.sort_values('end_time').traj_id.unique(), np.arange(len(stp.sort_values('end_time').traj_id.unique()))): order[name] = place
    #     stop_dept_order_df = stop_dept_order_df.append(order, ignore_index=True)

    print('Starting feature extraction for: '+sq_name+' in '+time_period)

    # smooth movement periods with a spline smoothing method
    smoothed_move_slice = Preprocessing.spline_smoothing(move_slice, s=3e1, UTM=True)

    # Smooth movement periods further with a rolling average 
    # for velocity analysis
    extra_smoothed_move_slice = Preprocessing.smooth_datasets(smoothed_move_slice, window=10)

    '''Velocity'''
    vel_dfs = VelocityFeats.get_velocities(extra_smoothed_move_slice, names)
    all_vel_mean = []
    all_vel_maxs = []
    all_vel_vars = []
    # loop through movement periods
    for veln in vel_dfs:
        # Average of velocity during the movement period
        all_vel_mean.append(veln.mean())
        # Maximum of velocity during the movement period
        all_vel_maxs.append(veln.max())
        # Variance of velocity during the movement period
        all_vel_vars.append(veln.var())
    # average across all movement periods
    vel_mean = pd.concat(all_vel_mean, axis=1).mean(axis=1)
    vel_maxs = pd.concat(all_vel_maxs, axis=1).mean(axis=1)
    vel_vars = pd.concat(all_vel_vars, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'vel_mean_'+time_period] = vel_mean
    Indiv_feats.loc[names, 'vel_maxs_'+time_period] = vel_maxs
    Indiv_feats.loc[names, 'vel_vars_'+time_period] = vel_vars
    # add to squad features df
    Squad_feats.loc[sq_name, 'vel_mean_'+time_period] = vel_mean.mean()
    Squad_feats.loc[sq_name, 'vel_maxs_'+time_period] = vel_maxs.mean()
    Squad_feats.loc[sq_name, 'vel_vars_'+time_period] = vel_vars.mean()


    '''Velocity-based features from soldier comparisons'''
    vel_vars, vel_diffs = VelocityFeats.get_vel_feats(vel_dfs)
    all_vel_var = []
    all_vel_diff = []
    # loop through movement periods
    for vvar, vdiff in zip(vel_vars, vel_diffs):
        # Average of velocity variance during the movement period
        all_vel_var.append(vvar.mean())
        # Maximum of velocity (max - min) during the movement period
        all_vel_diff.append(vdiff.mean())
    # average across all movement periods
    vel_var = np.mean(all_vel_var)
    vel_diff = np.mean(all_vel_diff)
    # add to indiv features df
    Indiv_feats.loc[names, 'vel_var_'+time_period] = vel_var
    Indiv_feats.loc[names, 'vel_diff_'+time_period] = vel_diff
    # add to squad features df
    Squad_feats.loc[sq_name, 'vel_var_'+time_period] = vel_var
    Squad_feats.loc[sq_name, 'vel_diff_'+time_period] = vel_diff


    '''Acceleration'''
    acc_dfs = VelocityFeats.get_accel(vel_dfs, names)
    all_acc_mean = []
    all_acc_maxs = []
    # loop through movement periods
    for accn in acc_dfs:
        # Average of acc during the movement period
        all_acc_mean.append(accn.mean())
        # Maximum of acc during the movement period
        all_acc_maxs.append(accn.max())
    # average across all movement periods
    acc_mean = pd.concat(all_acc_mean, axis=1).mean(axis=1)
    acc_maxs = pd.concat(all_acc_maxs, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'acc_mean_'+time_period] = acc_mean
    Indiv_feats.loc[names, 'acc_maxs_'+time_period] = acc_maxs
    # add to squad features df
    Squad_feats.loc[sq_name, 'acc_mean_'+time_period] = acc_mean.mean()
    Squad_feats.loc[sq_name, 'acc_maxs_'+time_period] = acc_maxs.mean()


    # '''Correlation of accelleration'''
    # acc_corr_dfs = acc_corr(acc_dfs, names=names, time_window=30)
    # all_acc_corr_mean = []
    # # loop through movement periods
    # for acorrn in acc_corr_dfs:
    #     # Average of acc during the movement period
    #     all_acc_corr_mean.append(acorrn.mean())
    # # average across all movement periods
    # acc_corr_mean = pd.concat(all_acc_corr_mean, axis=1).mean(axis=1)
    # # add to indiv features df
    # Indiv_feats.loc[names, 'acc_corr_mean_'+time_period] = acc_corr_mean
    # # add to squad features df
    # Squad_feats.loc[sq_name, 'acc_corr_mean_'+time_period] = acc_corr_mean.mean()


    # '''Correlation of velocity'''
    # vel_corr_dfs = vel_corr(vel_dfs, names=names, time_window=30)
    # all_vel_corr_mean = []
    # # loop through movement periods
    # for vcorrn in vel_corr_dfs:
    #     # Average of acc during the movement period
    #     all_vel_corr_mean.append(vcorrn.mean())
    # # average across all movement periods
    # vel_corr_mean = pd.concat(all_vel_corr_mean, axis=1).mean(axis=1)
    # # add to indiv features df
    # Indiv_feats.loc[names, 'vel_corr_mean_'+time_period] = vel_corr_mean
    # # add to squad features df
    # Squad_feats.loc[sq_name, 'vel_corr_mean_'+time_period] = vel_corr_mean.mean()



    '''Spatial Features'''
    print('Extracting spatial features from: '+sq_name+' in '+time_period)

    '''Stretch index'''
    cent_dists = SpatialFeats.get_cent_dist(move_slice) 
    all_cent_dist_mean = []
    all_cent_dist_meds = []
    all_cent_dist_maxs = []
    all_cent_dist_mins = []
    # loop thorugh movement periods
    for cent_dist in cent_dists:
        # Average of cent_dists during the movement period
        all_cent_dist_mean.append(cent_dist.mean())
        # Median of cent_dists during the movement period
        all_cent_dist_meds.append(cent_dist.median())
        # Max of cent_dists during the movement period
        all_cent_dist_maxs.append(cent_dist.max())
        # Min of cent_dists during the movement period
        all_cent_dist_mins.append(cent_dist.min())
    # average across all movement periods
    cent_dist_mean = pd.concat(all_cent_dist_mean, axis=1).mean(axis=1)
    cent_dist_meds = pd.concat(all_cent_dist_meds, axis=1).mean(axis=1)
    cent_dist_maxs = pd.concat(all_cent_dist_maxs, axis=1).mean(axis=1)
    cent_dist_mins = pd.concat(all_cent_dist_mins, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'cent_dist_mean_'+time_period] = cent_dist_mean
    Indiv_feats.loc[names, 'cent_dist_meds_'+time_period] = cent_dist_meds
    Indiv_feats.loc[names, 'cent_dist_maxs_'+time_period] = cent_dist_maxs
    Indiv_feats.loc[names, 'cent_dist_mins_'+time_period] = cent_dist_mins
    # add to squad features df (average indivs)
    Squad_feats.loc[sq_name, 'cent_dist_mean_'+time_period] = cent_dist_mean.mean()
    Squad_feats.loc[sq_name, 'cent_dist_meds_'+time_period] = cent_dist_meds.mean()
    Squad_feats.loc[sq_name, 'cent_dist_maxs_'+time_period] = cent_dist_maxs.mean()
    Squad_feats.loc[sq_name, 'cent_dist_mins_'+time_period] = cent_dist_mins.mean()
    

    '''Nearest neighbor distance'''
    neighbor_ds = SpatialFeats.neighbor_dists(move_slice)
    all_dist_mean = []
    all_dist_var = []
    # loop through movement periods
    for ds in neighbor_ds:
        # Average of nearest neighbor distance during the movement period
        all_dist_mean.append(ds.mean())
        # Varaiance nearest neighbor distance during the movement period
        all_dist_var.append(ds.var())
    # average across all movement periods
    dist_mean = pd.concat(all_dist_mean, axis=1).mean(axis=1)
    dist_var = pd.concat(all_dist_var, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'dist_mean_'+time_period] = dist_mean
    Indiv_feats.loc[names, 'dist_var_' +time_period] = dist_var
    # add to squad features df
    Squad_feats.loc[sq_name, 'dist_mean_'+time_period] = dist_mean.mean()
    Squad_feats.loc[sq_name, 'dist_var_' +time_period] = dist_var.mean()

    

    '''Surface area (hull)'''
    SAs = SpatialFeats.get_surface_area(move_slice) 
    all_SA_mean = []
    all_SA_meds = []
    all_SA_maxs = []
    all_SA_mins = []
    # loop thorugh movement periods
    for SAn in SAs:
        # Average of SA during the movement period
        all_SA_mean.append(SAn.mean())
        # Median of SA during the movement period
        all_SA_meds.append(SAn.median())
        # Max of SA during the movement period
        all_SA_maxs.append(SAn.max())
        # Min of SA during the movement period
        all_SA_mins.append(SAn.min())
    # average across all movement periods
    SA_mean = pd.concat(all_SA_mean, axis=1).mean(axis=1)
    SA_meds = pd.concat(all_SA_meds, axis=1).mean(axis=1)
    SA_maxs = pd.concat(all_SA_maxs, axis=1).mean(axis=1)
    SA_mins = pd.concat(all_SA_mins, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'SA_mean_'+time_period] = [SA_mean.values[0]]*len(names)
    Indiv_feats.loc[names, 'SA_meds_'+time_period] = [SA_meds.values[0]]*len(names)
    Indiv_feats.loc[names, 'SA_maxs_'+time_period] = [SA_maxs.values[0]]*len(names)
    Indiv_feats.loc[names, 'SA_mins_'+time_period] = [SA_mins.values[0]]*len(names)
    # add to squad features df
    Squad_feats.loc[sq_name, 'SA_mean_'+time_period] = SA_mean.values
    Squad_feats.loc[sq_name, 'SA_meds_'+time_period] = SA_meds.values
    Squad_feats.loc[sq_name, 'SA_maxs_'+time_period] = SA_maxs.values
    Squad_feats.loc[sq_name, 'SA_mins_'+time_period] = SA_mins.values


    '''Voronoi spaces'''
    voronoi_areas, voronoi_ratios = SpatialFeats.get_voronoi_areas(move_slice)
    all_VA_mean = []
    all_VR_mean = []
    # loop thorugh movement periods
    for VAn, VRn in zip(voronoi_areas, voronoi_ratios):
        # Average of Voronoi area during the movement period
        all_VA_mean.append(VAn.mean())
        # Median of Voronoi ratio during the movement period
        all_VR_mean.append(VRn.mean())
    # average across all movement periods
    VA_mean = pd.concat(all_VA_mean, axis=1).mean(axis=1)
    VR_mean = pd.concat(all_VR_mean, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'VorArea_mean_'+time_period] = VA_mean
    Indiv_feats.loc[names, 'VorRatio_mean_'+time_period] = VR_mean
    # add to squad features df
    Squad_feats.loc[sq_name, 'VorArea_mean_'+time_period] = VA_mean.mean()
    Squad_feats.loc[sq_name, 'VorRatio_mean_'+time_period] = VR_mean.mean()


    print('Extracting directional correlation features features from: '+sq_name+' in '+time_period)

    '''Directional correlation'''            
    # Leadership ranking, heirarchy and highly correlated segments
    window_length = 9
    time_delay_dfs, HCS_ratio_dfs, graphs = DirectionalCorrelation.get_directional_corr(smoothed_move_slice, names=names, UTM=True, threshold = 10, window_length=window_length)
    # graphs = pairs_directional_corr(smooth_move_slice, names=names, UTM=True, dist_threshold = 10, window_length=window_length)
    
    # # plotting
    # leadership_graph_ani(time_delay_dfs, graphs, names, sq_name, show=False)
    # leadership_plot_periods(time_delay_dfs, sq_name)

    # Leadership score
    mean_TD = pd.concat(time_delay_dfs).mean()
    # add to indiv features df
    Indiv_feats.loc[names, 'Leadership_'+time_period] = mean_TD
    # add to squad features df
    Squad_feats.loc[sq_name, 'Leadership_'+time_period] = mean_TD.mean()

    # HCS
    mean_HCS = pd.concat(HCS_ratio_dfs).mean()
    # add to indiv features df
    Indiv_feats.loc[names, 'HCS_'+time_period] = mean_HCS
    # add to squad features df
    Squad_feats.loc[sq_name, 'HCS_'+time_period] = mean_HCS.mean()

    # Leadership Graph consistency
    G_consist, G_adj_consist = DirectionalCorrelation.dir_corr_graph_comparison(graphs)
    # add to indiv features df
    Indiv_feats.loc[names, 'lead_consist_'+time_period] = G_consist
    # add to squad features df
    Squad_feats.loc[sq_name, 'lead_consist_'+time_period] = G_consist
    # add to indiv features df
    Indiv_feats.loc[names, 'lead_consist_adj_'+time_period] = G_adj_consist.mean()
    # add to squad features df
    Squad_feats.loc[sq_name, 'lead_consist_adj_'+time_period] = G_adj_consist.mean().mean()



    '''Break metrics'''
    # Rest count
    # add to indiv features df
    Indiv_feats.loc[names, 'Rest_Count_'+time_period] = len(rest)
    # add to squad features df
    Squad_feats.loc[sq_name, 'Rest_Count_'+time_period] = len(rest)
    # Rest time
    if rest:
        rest_times = []
        for r in rest:
            rest_times.append(len(r))
        rest_time = np.sum(rest_times)
    else:
        rest_time=0
    # add to indiv features df
    Indiv_feats.loc[names, 'Rest_time_'+time_period] = rest_time
    # add to squad features df
    Squad_feats.loc[sq_name, 'Rest_time_'+time_period] = rest_time

    # Rest percent of time
    move_times = []
    for m in move_slice:
        move_times.append(len(m))
    move_time = np.sum(move_times)
    rest_ratio = rest_time/move_time
    # add to indiv features df
    Indiv_feats.loc[names, 'Rest_percent_'+time_period] = rest_ratio
    # add to squad features df
    Squad_feats.loc[sq_name, 'Rest_percent_'+time_period] = rest_ratio


    '''Re-orient data'''
    # Orient ruck periods
    move_slices_oriented = PACS.PACS_transform(smoothed_move_slice)


    '''Distribution consistency'''
    # F-test
    X_ftest, Y_ftest = PacsFeats.dist_consistency_Ftest(move_slices_oriented, names=names)
    # add to indiv features df
    Indiv_feats.loc[names, 'Ftest_X_mean_'+time_period] = X_ftest
    Indiv_feats.loc[names, 'Ftest_Y_mean_'+time_period] = Y_ftest
    # add to squad features df
    Squad_feats.loc[sq_name, 'Ftest_X_mean_'+time_period] = X_ftest.mean()
    Squad_feats.loc[sq_name, 'Ftest_Y_mean_'+time_period] = Y_ftest.mean()

    # Wasserstein distance
    X_wass_df, Y_wass_df = PacsFeats.dist_consistency_wasserstein(move_slices_oriented, names=names)
    # add to indiv features df
    Indiv_feats.loc[names, 'Wass_X_mean_'+time_period] = X_wass_df
    Indiv_feats.loc[names, 'Wass_Y_mean_'+time_period] = Y_wass_df
    # add to squad features df
    Squad_feats.loc[sq_name, 'Wass_X_mean_'+time_period] = X_wass_df.mean()
    Squad_feats.loc[sq_name, 'Wass_Y_mean_'+time_period] = Y_wass_df.mean()


    

    # if time_period=='whole':
    #     # # prep dfs for plotting
    #     # ruck_oriented_prepped = prep_df([test_data], change_units=False)
    #     # # plot these dfs
    #     # joint_subplots(ruck_oriented_prepped, move_slice[0].attrs['name'], move_slice, rest=False, colormap = color_dictionary)
    #     ruck_oriented_prepped = PACS.prep_df(move_slices_oriented, change_units=False)
    #     # plot these dfs
    #     PACS.joint_subplots(ruck_oriented_prepped, 'orient_test', [smooth_data[1]], rest=False, colormap = color_dictionary)


    
    '''Length/width ratio'''
    LW_ratios = PacsFeats.LW_ratio(move_slices_oriented)
    all_LW_mean = []
    all_LW_maxs = []
    all_LW_mins = []
    # loop through movement periods
    for LWn in LW_ratios:
        # Average of LW ratio during the movement period
        all_LW_mean.append(LWn.mean())
        # Maximum of LW ratio during the movement period
        all_LW_maxs.append(LWn.max())
        # Minumum of LW ratio during the movement period
        all_LW_mins.append(LWn.min())
    # average across all movement periods
    LW_mean = np.mean(all_LW_mean)
    LW_maxs = np.mean(all_LW_maxs)
    LW_mins = np.mean(all_LW_mins)
    # add to indiv features df
    Indiv_feats.loc[names, 'LW_mean_'+time_period] = LW_mean
    Indiv_feats.loc[names, 'LW_maxs_'+time_period] = LW_maxs
    Indiv_feats.loc[names, 'LW_mins_'+time_period] = LW_mins
    # add to squad features df
    Squad_feats.loc[sq_name, 'LW_mean_'+time_period] = LW_mean
    Squad_feats.loc[sq_name, 'LW_maxs_'+time_period] = LW_maxs
    Squad_feats.loc[sq_name, 'LW_mins_'+time_period] = LW_mins



    '''Nearest neighbor distance'''
    x_neighbors, y_neighbors = PacsFeats.get_neighbor_dists(move_slices_oriented, names=names)
    all_Xnn_mean = []
    all_Ynn_mean = []
    # loop through movement periods
    for Xn, Yn in zip(x_neighbors, y_neighbors):
        # Average of X nearest neighbor distance during the movement period
        all_Xnn_mean.append(Xn.mean())
        # Average of Y nearest neighbor during the movement period
        all_Ynn_mean.append(Yn.mean())
    # average across all movement periods
    Xnn_mean = pd.concat(all_Xnn_mean, axis=1).mean(axis=1)
    Ynn_mean = pd.concat(all_Ynn_mean, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'Xnn_mean_'+time_period] = Xnn_mean
    Indiv_feats.loc[names, 'Ynn_mean_'+time_period] = Ynn_mean
    # add to squad features df
    Squad_feats.loc[sq_name, 'Xnn_mean_'+time_period] = Xnn_mean.mean()
    Squad_feats.loc[sq_name, 'Ynn_mean_'+time_period] = Ynn_mean.mean()
    



    '''Spatial exploration index'''
    SEIs = PacsFeats.get_SEIs(move_slices_oriented, names=names)
    all_SEI_mean = []
    all_SEI_maxs = []
    # loop through movement periods
    for SEIn in SEIs:
        # Average of SEI during the movement period
        all_SEI_mean.append(SEIn.mean())
        # Maximum of SEI during the movement period
        all_SEI_maxs.append(SEIn.max())
    # average across all movement periods
    SEI_mean = pd.concat(all_SEI_mean, axis=1).mean(axis=1)
    SEI_maxs = pd.concat(all_SEI_maxs, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'SEI_mean_'+time_period] = SEI_mean
    Indiv_feats.loc[names, 'SEI_maxs_'+time_period] = SEI_maxs
    # add to squad features df
    Squad_feats.loc[sq_name, 'SEI_mean_'+time_period] = SEI_mean.mean()
    Squad_feats.loc[sq_name, 'SEI_maxs_'+time_period] = SEI_maxs.mean()


    '''Regularity features'''
    print('Extracting regularity features features from: '+sq_name+' in '+time_period)


    '''VAR model'''
    VAR_errs = RegularityFeats.VAR_model(move_slices_oriented, names)
    all_VAR_mean = []
    for VARerr in VAR_errs:
        # Average of VAR error the movement period
        all_VAR_mean.append(VARerr.mean())
    # average across all movement periods
    VAR_mean = pd.concat(all_VAR_mean, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'VAR_mean_'+time_period] = VAR_mean
    # add to squad features df
    Squad_feats.loc[sq_name, 'VAR_mean_'+time_period] = VAR_mean.mean()

    '''VARX model'''
    VARX_errs = RegularityFeats.VARX_model(move_slices_oriented, names)
    all_VARX_mean = []
    for VARXerr in VARX_errs:
        # Average of VAR error the movement period
        all_VARX_mean.append(VARXerr.mean())
    # average across all movement periods
    VARX_mean = pd.concat(all_VARX_mean, axis=1).mean(axis=1)
    # add to indiv features df
    Indiv_feats.loc[names, 'VARX_mean_'+time_period] = VARX_mean
    # add to squad features df
    Squad_feats.loc[sq_name, 'VARX_mean_'+time_period] = VARX_mean.mean()


    '''Entropy of PACS locations'''
    entropy_df = RegularityFeats.PACS_entropy(move_slices_oriented, names)
    # average across all movement periods
    all_entropy = entropy_df.mean()
    # add to indiv features df
    Indiv_feats.loc[names, 'PACS_entropy_'+time_period] = all_entropy
    # add to squad features df
    Squad_feats.loc[sq_name, 'PACS_entropy_'+time_period] = all_entropy.mean()

    '''Entropy of kinematic metrics'''
    # using different PDF (normalized histogram) parameters for each feature i.e. bion count and size 

    vel_ent = RegularityFeats.time_series_metric_entropy(vel_dfs, range=[0,5], bins=100)
    # add to indiv features df
    Indiv_feats.loc[names, 'vel_ent_'+time_period] = vel_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'vel_ent_'+time_period] = vel_ent.mean()

    vel_var_ent = RegularityFeats.time_series_metric_entropy(vel_vars, range=[0,1], bins=1000)
    # add to indiv features df
    Indiv_feats.loc[names, 'vel_var_ent_'+time_period] = vel_var_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'vel_var_ent_'+time_period] = vel_var_ent

    vel_diffs_ent = RegularityFeats.time_series_metric_entropy(vel_diffs, range=[0,5], bins=100)
    # add to indiv features df
    Indiv_feats.loc[names, 'vel_diffs_ent_'+time_period] = vel_diffs_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'vel_diffs_ent_'+time_period] = vel_diffs_ent

    acc_dfs_ent = RegularityFeats.time_series_metric_entropy(acc_dfs, range=[-1,1], bins=100)
    # add to indiv features df
    Indiv_feats.loc[names, 'acc_dfs_ent_'+time_period] = acc_dfs_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'acc_dfs_ent_'+time_period] = acc_dfs_ent.mean()

    cent_dists_ent = RegularityFeats.time_series_metric_entropy(cent_dists, range=[0,500], bins=1000)
    # add to indiv features df
    Indiv_feats.loc[names, 'cent_dists_ent_'+time_period] = cent_dists_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'cent_dists_ent_'+time_period] = cent_dists_ent.mean()

    neighbor_ds_ent = RegularityFeats.time_series_metric_entropy(neighbor_ds, range=[0,1000], bins=5000)
    # add to indiv features df
    Indiv_feats.loc[names, 'neighbor_ds_ent_'+time_period] = neighbor_ds_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'neighbor_ds_ent_'+time_period] = neighbor_ds_ent.mean()

    SAs_ent = RegularityFeats.time_series_metric_entropy(SAs, range=[0,1000], bins=1000)
    # add to indiv features df
    Indiv_feats.loc[names, 'SAs_ent_'+time_period] = SAs_ent[0]
    # add to squad features df
    Squad_feats.loc[sq_name, 'SAs_ent_'+time_period] = SAs_ent[0]

    voronoi_areas_ent = RegularityFeats.time_series_metric_entropy(voronoi_areas, range=[0,1000], bins=1000)
    # add to indiv features df
    Indiv_feats.loc[names, 'voronoi_areas_ent_'+time_period] = voronoi_areas_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'voronoi_areas_ent_'+time_period] = voronoi_areas_ent.mean()

    voronoi_ratios_ent = RegularityFeats.time_series_metric_entropy(voronoi_ratios, range=[0,1], bins=100)
    # add to indiv features df
    Indiv_feats.loc[names, 'voronoi_ratios_ent_'+time_period] = voronoi_ratios_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'voronoi_ratios_ent_'+time_period] = voronoi_ratios_ent.mean()

    LW_ratios_ent = RegularityFeats.time_series_metric_entropy(LW_ratios, range=[0,100], bins=1000)
    # add to indiv features df
    Indiv_feats.loc[names, 'LW_ratios_ent_'+time_period] = LW_ratios_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'LW_ratios_ent_'+time_period] = LW_ratios_ent

    x_neighbors_ent = RegularityFeats.time_series_metric_entropy(x_neighbors, range=[0,50], bins=100)
    # add to indiv features df
    Indiv_feats.loc[names, 'x_neighbors_ent_'+time_period] = x_neighbors_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'x_neighbors_ent_'+time_period] = x_neighbors_ent.mean()

    y_neighbors_ent = RegularityFeats.time_series_metric_entropy(y_neighbors, range=[0,300], bins=1000)
    # add to indiv features df
    Indiv_feats.loc[names, 'y_neighbors_ent_'+time_period] = y_neighbors_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'y_neighbors_ent_'+time_period] = y_neighbors_ent.mean()

    SEIs_ent = RegularityFeats.time_series_metric_entropy(SEIs, range=[0,250], bins=1000)
    # add to indiv features df
    Indiv_feats.loc[names, 'SEIs_ent_'+time_period] = SEIs_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'SEIs_ent_'+time_period] = SEIs_ent.mean()

    VAR_errs_ent = RegularityFeats.time_series_metric_entropy(VAR_errs, range=[0,15], bins=150)
    # add to indiv features df
    Indiv_feats.loc[names, 'VAR_errs_ent_'+time_period] = VAR_errs_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'VAR_errs_ent_'+time_period] = VAR_errs_ent.mean()

    VARX_errs_ent = RegularityFeats.time_series_metric_entropy(VARX_errs, range=[0,15], bins=150)
    # add to indiv features df
    Indiv_feats.loc[names, 'VARX_errs_ent_'+time_period] = VARX_errs_ent
    # add to squad features df
    Squad_feats.loc[sq_name, 'VARX_errs_ent_'+time_period] = VARX_errs_ent.mean()



    '''Positioning Graphs'''
    # positioning_graphs = dyad_movement_graph(move_slice, move_slices_oriented, names, color_dictionary, sq_name, show=False)



    # '''Doctrine Metrics'''
    # dict_dist_ratios = RegularityFeats.doct_dists(move_slice, names=names, UTM=True)
    # # average across all movement periods
    # all_dist_ratios = pd.concat(dict_dist_ratios, axis=1).mean(axis=1)
    # # add to indiv features df
    # Indiv_feats.loc[names, 'doct_dist_ratio_'+time_period] = all_dist_ratios
    # # add to squad features df
    # Squad_feats.loc[sq_name, 'doct_dist_ratio_'+time_period] = all_dist_ratios.mean()

    # doct_vel_ratios = doct_vels(vel_dfs, names=names)
    # # average across all movement periods
    # all_vel_ratios = pd.concat(doct_vel_ratios, axis=1).mean(axis=1)
    # # add to indiv features df
    # Indiv_feats.loc[names, 'doct_vel_ratio_'+time_period] = all_vel_ratios
    # # add to squad features df
    # Squad_feats.loc[sq_name, 'doct_vel_ratio_'+time_period] = all_vel_ratios.mean()

    # if rest:
    #     inter_break_variance, break_duration_variance = break_timings(move_slice, rest)
    #     # add to indiv features df
    #     Indiv_feats.loc[names, 'inter_break_var_'+time_period] = inter_break_variance
    #     # add to squad features df
    #     Squad_feats.loc[sq_name, 'inter_break_var_'+time_period] = inter_break_variance
    #     # add to indiv features df
    #     Indiv_feats.loc[names, 'break_duration_var_'+time_period] = break_duration_variance
    #     # add to squad features df
    #     Squad_feats.loc[sq_name, 'break_duration_var_'+time_period] = break_duration_variance


    
    print('Extracting cluster features from: '+sq_name+' in '+time_period)


    '''Clustering features'''
    # apply and get metrics from a few clustering methods
    for epsilon in [5, 10, 25, 50]:
        for method in ['HDBSCAN', 'DBSCAN']:

            all_inertias, all_labels, all_scores = ClusteringFeats.cluster_for_separation(move_slice, method=method, epsilon=epsilon)

            # '''Cluster consistency'''
            # knee_vals = clust_consistency(all_labels)
            # # add to indiv features df
            # Indiv_feats.loc[names, method + '_' + str(epsilon) + '_clust_knee_'+time_period] = np.mean(knee_vals)
            # # add to squad features df
            # Squad_feats.loc[sq_name, method + '_' + str(epsilon) + '_clust_knee_'+time_period] = np.mean(knee_vals)


            '''Outlier time'''
            # get outlier times
            outlier_times = ClusteringFeats.get_outlier_time(all_labels)
            # sum across all movement periods
            outlier_time_sum = pd.concat(outlier_times, axis=1).sum(axis=1)
            # add to indiv features df
            Indiv_feats.loc[names, method + '_' + str(epsilon) + '_outlier_time_sum_'+time_period] = outlier_time_sum
            # add to squad features df
            Squad_feats.loc[sq_name, method + '_' + str(epsilon) + '_outlier_time_sum_'+time_period] = outlier_time_sum.mean()


            '''Membership confidence'''
            if method=='HDBSCAN':
                all_confidence = []
                # loop through movemement periods
                for inertia in all_inertias:
                    all_confidence.append(inertia.mean())
                # average across all movement periods
                membership_mean = pd.concat(all_confidence, axis=1).mean(axis=1)
                # add to indiv features df
                Indiv_feats.loc[names, method + '_' + str(epsilon) + '_membership_mean_'+time_period] = membership_mean
                # add to squad features df
                Squad_feats.loc[sq_name, method + '_' + str(epsilon) + '_membership_mean_'+time_period] = membership_mean.mean()




Indiv_feats.to_pickle(os.getcwd() + '\\SampleFeatures\\Indiv_feats.pkl') 
Squad_feats.to_pickle(os.getcwd() + '\\SampleFeatures\\Squad_feats.pkl') 

