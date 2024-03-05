'''
For extracting features from oriented and segmented data

Using different clustering methods and extracting features from each

'''
# imports 


from sklearn.cluster import KMeans, DBSCAN
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import math
from tqdm import tqdm
import numpy as np
from kneed.knee_locator import KneeLocator
import plotly.express as px
from hdbscan import HDBSCAN
import pandas as pd
import io
import PIL
import os




def cluster_for_separation(datasets, UTM=True, method=None, epsilon=None):
    """
    CLuster the data for separation metrics, using teh clustering method of choice

    Args:
        datasets (list): list of movement period DataFrames
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.
        method (clustering method (DBSCAN or HDBSCAN), optional): clustering method. Defaults to None.
        epsilon (float, optional): epsilon for clustering method, if applicable. Defaults to None.

    Returns:
        _type_: _description_
        all_inertias (list) : list of dataframes with cluster membership probabilities for each soldier, if method = 'HDBSCAN'
        all_labels: (list): list of dataframes with cluster labels for each soldier at each timepoint
        all_scores (list): list of series the silouette scores for each timepoint
    """
    '''
    find 2 k-means clusters each timepoint
    threshold separation metric and find where quads are split

    if this doesnt work then use another clustering metric as well and threshold both
    '''

    print('Extracting separation metrics')

    # initialize lists
    all_inertias = []
    all_labels = []
    all_scores = []
    all_ks = []

    # loop through datasets
    # for data in tqdm(datasets):
    for data in datasets:

        # get na,es
        names = []
        for col in data.columns:
            # names.append(col.split()[0])
            names.append(col[1])
        # get unique names and drop 'unnamed'
        if '' in names:
            print('getting rid of empty name')
            names = [x for x in list(set(names)) if 'Unnamed:' not in x][1:]
        else:
            names = [x for x in list(set(names)) if 'Unnamed:' not in x]

        # initialize lists
        inertias_data = []
        labels_data = []
        scores_data = []
        all_ks_data = []

        # loop through timepoints and find the distance between 
        # data.dropna(inplace=True) # Drop the last timepoints if all soldiers not included            
        # for idx in tqdm(data.index):

        for idx in data.index:

            # get points for kmeans
            points=[]

            # loop through names
            for n in names:

                # choose units 
                if UTM:
                    points.append([data['UTM_x', n][idx], data['UTM_y', n][idx]])
                else:
                    points.append([data['longitude', n][idx]*111139, data['latitude', n][idx]*111139])

            # If any nan, clustering will fail, append nan value if true
            if pd.DataFrame(points).isna().any().any():
                inertias_data.append([np.nan]*len(names))
                labels_data.append([np.nan]*len(names))
                scores_data.append(np.nan)
            else:

                # if HDDBSCAN get intertias data
                if method == 'HDBSCAN':
                    
                    # initialize model
                    fitted = HDBSCAN(min_cluster_size=2, allow_single_cluster=True,
                                    cluster_selection_epsilon=epsilon, min_samples=2, )
                    
                    # fit to data
                    fitted.fit(points)

                    # append probabilities (HDBSCAN membership probability)
                    inertias_data.append(fitted.probabilities_)

                # if DBSCAN append nan value to porbabilities
                elif method == 'DBSCAN':
                    
                    # initialize model
                    fitted = DBSCAN(eps = epsilon, min_samples = 2, )

                    # fit to data
                    fitted.fit(points)

                    # append nan porbabilities
                    inertias_data.append([np.nan]*len(points))

                # attempt to normalize squad labels, forcing the first soldier to be in squad 0 always
                if fitted.labels_[0]==1:
                    label_changer = {0:1, 1:0}
                    new_labels = pd.Series(fitted.labels_).replace(label_changer).to_numpy()
                elif fitted.labels_[0]==2:
                    label_changer = {0:2, 2:0}
                    new_labels = pd.Series(fitted.labels_).replace(label_changer).to_numpy()
                elif fitted.labels_[0]==3:
                    label_changer = {0:3, 3:0}
                    new_labels = pd.Series(fitted.labels_).replace(label_changer).to_numpy()
                elif fitted.labels_[0]==4:
                    label_changer = {0:4, 4:0}
                    new_labels = pd.Series(fitted.labels_).replace(label_changer).to_numpy()
                else:
                    new_labels = fitted.labels_

                # replace with new labels
                labels_data.append(new_labels)

                # get silouette score
                if max(new_labels) <= 0:
                    scores_data.append(np.nan)
                else:
                    scores_data.append(silhouette_score(points, new_labels))

        # append resuilts to lists
        all_inertias.append(pd.DataFrame(inertias_data, columns=names))
        all_labels.append(pd.DataFrame(labels_data, columns=names))
        all_scores.append(pd.Series(scores_data))

    return all_inertias, all_labels, all_scores



def DBSCAN_for_separation(datasets):
    '''
    find 2 k-means clusters each timepoint
    threshold separation metric and find where quads are split

    if this doesnt work then use another clustering metric as well and threshold both
    '''

    print('Extracting separation metrics')

    all_cores = []
    all_labels = []
    all_comps = []

    # for data in tqdm(datasets):
    for data in datasets:

        # if 'PLT2SQ3' in data.attrs['name']:
        
        #     import pdb
        #     pdb.set_trace()

        names = []
        for col in data.columns:
            # names.append(col.split()[0])
            names.append(col[1])
        # get unique names and drop 'unnamed'
        # names = [x for x in list(set(names)) if 'Unnamed:' not in x][1:]
        if '' in names:
            print('getting rid of empty name')
            names = [x for x in list(set(names)) if 'Unnamed:' not in x][1:]
        else:
            names = [x for x in list(set(names)) if 'Unnamed:' not in x]

        cores_data = []
        labels_data = []
        comps_data = []

        # loop through timepoints and find the distance between 
        data.dropna(inplace=True) # Drop the last timepoints if all soldiers not included            
        # for idx in tqdm(data.index):
        for idx in data.index:

            # get points for kmeans
            points=[]
            for n in names:
                # points.append([data[n+' longitude'][idx], data[n+' latitude'][idx]])
                points.append([data['longitude', n][idx] *111139, data['latitude', n][idx] *111139])


            # use DBSCAN here

            clusts = DBSCAN(eps = 25, min_samples = 2)
            clusts.fit(points)

            cores_data.append(clusts.core_sample_indices_)
            labels_data.append(clusts.labels_)
            comps_data.append(clusts.components_)
            

            # distance = math.dist(fitted.cluster_centers_[0], fitted.cluster_centers_[1])
            # groupsize = min(len(np.where(fitted.labels_)[0]), len(np.where(fitted.labels_-1)[0]))
            # distances.append(distance)
            # g_sizes.append(groupsize)

        # separation_metrics.append(pd.Series(distances, name=ruck.attrs['name']))
        # min_group_sizes.append(pd.Series(g_sizes, name=ruck.attrs['name']))

        # import pdb
        # pdb.set_trace()

        # test = pd.DataFrame(labels_data, columns=names)

        # import itertools
        # marker = itertools.cycle(('v', '+', '.', 'o', '*', '1')) 

        # for c in test.columns:
        #     plt.plot(test[c], label=c, marker=next(marker))
        
        # plt.legend()
        # plt.show()

        all_cores.append(pd.Series(cores_data))
        all_labels.append(pd.DataFrame(labels_data, columns=names))
        all_comps.append(pd.Series(comps_data))
    
    # fig = plt.figure() 

    # for count, (separation, min_group) in enumerate(zip(separation_metrics, min_group_sizes)):


    return all_cores, all_labels, all_comps



def prep_cluster_df(datasets, all_labels, change_units=True, decimate=0):
    '''
    Prep dataframe for plotting
    
    Input: dataframe of location data (columns= '[MASRTE ID] longitude' and latitude)
    
    Output: Long form dataframe for seaborn plots (columns=['longitude','latitude','ID','time'])
    
    '''
    prepped_clust_dfs = []

    print('re-formatting data')

    for df, labels in zip(tqdm(datasets), all_labels):

        
        # # decimate the signal (take every X samples) if decimate doesnt = 0 
        # if decimate:
        #     ndf = df[::decimate]
        #     ndf.attrs['name'] = df.attrs['name']
        #     df = ndf

        # get unique names from columns
        names = []
        for col in df.columns:
            if type(col) == tuple:
                names.append(col[1])
            else:
                names.append(col.split()[0])
        # get unique names and drop 'unnamed'
        names = [x for x in list(set(names)) if 'Unnamed:' not in x]

        # initialize dfs
        new_df = pd.DataFrame(columns=['longitude','latitude','ID','time', 'X_err', 'Y_err', 'cluster'])

        # get individual soldier data
        indiv_dfs = []
        for ID in names:
            # separate This soldier
            if type(col) == tuple:
                indiv_data = df[[c for c in df.columns if np.logical_and(ID in c, np.logical_or('longitude' in c, 'latitude' in c))]]
                indiv_data.reset_index(inplace=True, drop=True)
            else:
                indiv_data = df[[c for c in df.columns if ID in c]]
            indiv_data.columns = ['latitude','longitude']
            # X_err = indiv_data['longitude'].rolling(60, center=True).var()*3*2
            # Y_err = indiv_data['latitude']. rolling(60, center=True).var()*3*2

            # normalize here
            indiv_data['longitude'] = indiv_data['longitude'] - df['longitude'].mean(axis=1).values
            indiv_data['latitude'] = indiv_data['latitude'] - df['latitude'].mean(axis=1).values
            
            # indiv_data = pd.concat([indiv_data, pd.Series(X_err,name='X_err')], axis=1)
            # indiv_data = pd.concat([indiv_data, pd.Series(Y_err,name='Y_err')], axis=1)

            # indiv_data = pd.concat([indiv_data, pd.Series([ID]*len(indiv_data),name='ID')], ignore_index=True)
            indiv_data['ID'] = [ID]*len(indiv_data)
            indiv_data['cluster'] = 'Cluster: ' + labels[ID].astype(str)
            # indiv_data['cluster'] = labels[ID]
            indiv_data.reset_index(names='time', inplace=True)

            indiv_dfs.append(indiv_data) 

            indiv_data.cluster.unique()
        
        new_df = pd.concat(indiv_dfs)

        # Rotate 90 degrees, right is forward in new_df
        # also convert to feet
        front_df = pd.DataFrame()
        # front_df['X'] = new_df['latitude'] * 111139 #(10000 / 90) * 3280
        # front_df['Y'] = -new_df['longitude'] * 111139 * math.cos(df[[c for c in df.columns if 'latitude' in c]].mean().mean())#(10000 / 90) * 3280
        if change_units:
            front_df['X'] = new_df['longitude'] * 111139  * math.cos(df[[c for c in df.columns if 'latitude' in c]].mean().mean())
            front_df['Y'] = new_df['latitude'] * 111139
            # front_df['X_err'] = new_df['X_err'] * 111139  * math.cos(df[[c for c in df.columns if 'latitude' in c]].mean().mean())
            # front_df['Y_err'] = new_df['Y_err'] * 111139
        else:
            front_df['X'] = new_df['longitude']
            front_df['Y'] = new_df['latitude'] 
            # front_df['X_err'] = new_df['X_err']
            # front_df['Y_err'] = new_df['Y_err']
        front_df['time'] = new_df['time']
        front_df['ID'] = new_df['ID']
        front_df['cluster'] = new_df['cluster']
        front_df.attrs['name'] = df.attrs['name']
        front_df['X_err'] = front_df['X'].rolling(30, center=True).var()/2
        front_df['Y_err'] = front_df['Y'].rolling(30, center=True).var()/2

        prepped_clust_dfs.append(front_df)

    return prepped_clust_dfs 



def make_cluster_gifs(prepped_clust_dfs):
    '''
    make gifs from plot_prepped datasets
    '''

    for count, df in enumerate(tqdm(prepped_clust_dfs)):

        
        m = 60

        print('trimming timepoints')

        # Trim timepoints
        df_name = df.attrs['name']
        time = df['time']
        dfn = []
        for t in time[::m]:
            df_t = df[df['time']==t]
            dfn.append(df_t)
        df = pd.concat(dfn)
        df.sort_values('time', inplace=True)
        df['time (min)'] = df['time']/60
        df['time (min)'] = df['time (min)'].astype(float).round(3)
        df.attrs['name'] = df_name


        # df['cluster'] = df['cluster'].astype(str)
        
        print(df.cluster.unique())

        # import pdb
        # pdb.set_trace()

        print('initialize animation plot')

        # sample plotly animated figure
        fig = px.scatter(df, x="X", y="Y", animation_frame="time (min)", animation_group="ID",
                color="cluster", title=df.attrs['name']+"_"+str(count),
                # range_y=[df['Y'].min(), df['Y'].max()], range_x=[df['Y'].min(), df['Y'].max()],#, category_orders = np.arange(len(df.cluster.unique()))) 
                #range_x=[df['X'].min(), df['X'].max()])
                # error_x='X_max', error_x_minus='X_min', error_y='Y_max', error_y_minus='Y_min')
                # error_x='X_err', error_y='Y_err', color_discrete_sequence=sns.color_palette(as_cmap=True))
                marginal_x='box', marginal_y='box')
        fig.update_xaxes(scaleanchor="y", scaleratio=1, dtick=10)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, dtick=10)

        print('generate animation plot')

        # generate images for each step in animation
        frames = []
        for s, fr in enumerate(tqdm(fig.frames)):
            # set main traces to appropriate traces within plotly frame
            fig.update(data=fr.data)
            # move slider to correct place
            fig.layout.sliders[0].update(active=s)
            # generate image of current state
            frames.append(PIL.Image.open(io.BytesIO(fig.to_image(format="png", scale = 2))))
        
        print('save plot')

        # create animated GIF
        frames[0].save(
                os.getcwd() + '\\Figures\\GIF_' + df.attrs['name']+"_"+str(count)+".gif",
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=90,
                loop=0            
            )
        print('GIF saved to: ' + os.getcwd() + '\\Figures\\GIF_' + df.attrs['name']+"_"+str(count)+".gif")

    return None



def get_group_names(ruck_slices, all_scores, all_labels):
    '''
    Get names of groups and outliers from cluster labels and non-nan sillouette scores
    '''
    # Identify and separate movement period groupings
    outlier_names = []
    grouped_names = []
    for this_squad_ruck, scores, labels in zip(ruck_slices, all_scores, all_labels):
        this_sq_groups = []
        this_sq_outliers = []
        for this_movement_period in this_squad_ruck:
            # get labels for this period
            # this_period_scores = scores[this_movement_period.index]
            this_period_labels = labels.iloc[this_movement_period.index]
            # find the most common groups
            groups = []
            for index, row in this_period_labels.iterrows():
                groups.append([row_name for (x, row_name) in zip(row,row.index) if x == 0])
                groups.append([row_name for (x, row_name) in zip(row,row.index) if x == 1]) 
                groups.append([row_name for (x, row_name) in zip(row,row.index) if x == 2]) 
                groups.append([row_name for (x, row_name) in zip(row,row.index) if x == 3]) 
                groups.append([row_name for (x, row_name) in zip(row,row.index) if x == 4]) 
            groups = [x for x in groups if x]
            # count how many for each group
            group_counts = defaultdict(int)
            for group in groups:
                group_str = '-'.join(sorted(group))  # sort the names to create a unique string for each group
                group_counts[group_str] += 1
            # sort by largest count
            sorted_d = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)

            names = this_movement_period['longitude'].columns.to_list()
            # labeled_names =[]
            this_period_group = []
            # Add cluster to this_sq_groups if together for at least 30% of movement period
            for d in sorted_d:
                if all(n in d[0].split('-') for n in names):
                    continue
                print(d)
                if d[1]>(len(this_movement_period)/3):
                    this_period_group.append(d[0].split('-'))

                # this_period_group.append(d[0].split('-'))
                # for x in d[0].split('-'):
                #     labeled_names.append(x)
                #     if len([n for n in names if n in labeled_names])==len(names):
                #         break
                # if len([n for n in names if n in labeled_names])==len(names):
                #     break
            this_sq_groups.append(this_period_group)
        grouped_names.append(this_sq_groups)
                        
                




        #     if len(np.where(this_period_scores.isna() == False)[0]) > len(this_period_scores)*0.3:
        #         labels_during_split = this_period_labels[this_period_scores.isna() == False]
        #         # ID groups
        #         this_period_group = [sold for sold in labels_during_split if labels_during_split[sold].median()>0]
        #         # group2 = [sold for sold in labels_during_split if not sold in group1]
        #     this_sq_groups.append(this_period_group)
        #     # find outliers
        #     this_period_outliers = []
        #     for this_soldier_labels in this_period_labels:
        #         # find where outliers are marked (more than 30% of period)
        #         if len(np.where(this_period_labels[this_soldier_labels] == -1)[0]) > len(this_period_scores)*0.3:
        #             this_period_outliers.append(this_soldier_labels)
        #     this_sq_outliers.append(this_period_outliers)
        # outlier_names.append(this_sq_outliers)
        # grouped_names.append(this_sq_groups)
    
    return grouped_names#, outlier_names



if __name__ == '__main__':
    # Initialize path to data
    # data_dir = os.getcwd() + '\\Data\\csv'
    # # Load datasets
    # raw_datasets = load_data(data_dir)
    # # Re-shape datasets
    # datasets = pivot_datsets(raw_datasets)
    # # Get centroids
    # centroids = get_centroid(datasets)

    # # for data in datasets:
    # #     if 'PLT2SQ3' in data.attrs['name']:
    # #         test = [data[:3000], data[5000:6000], data[7000:8000]]
    # #         test[0].attrs['name'] = 'test'
    # #         test[1].attrs['name'] = 'test'
    # #         test[2].attrs['name'] = 'test'

    # # cluster before re-orientation
    # all_cores, all_labels, all_comps = DBSCAN_for_separation(datasets)

    # # prepped_clust_dfs = prep_cluster_df(test, all_labels)

    # # make_cluster_gifs(prepped_clust_dfs)

    # # plt.plot(all_scores[0], label='silouette scores')
    # # plt.hlines(.8, 0, 1000, label='threshold')
    # # plt.legend()
    # # plt.show()

    # fig, ax = plt.subplots(len(all_labels),1) 
    # for count, (labels, datas, timing_slices) in enumerate(zip(all_scores, datasets, ruck_slices)):
    #     # for cnt, c in enumerate(labels.columns):
    #     #     ax[count].plot(labels[c]+(0.1*cnt)-(.4))
    #     ax[count].set_ylabel(datas.attrs['name'], rotation='vertical')
    #     ax[count].plot(pd.DataFrame([l for l in labels]).mean(axis=1))
    #     # ax[count].hlines([0.6], 0, len(labels))
    #     ax[count].set_xticks(np.arange(len(labels))[::600], np.arange(len(labels))[::600]/60)
    #     # ax[count].set_yticks([-1, 0, 1, 2], ['outlier', 'cluster 1', 'cluster 2', 'cluster 3'])
    #     # ax[count].hlines([-1.5,-.5, .5, 1.5, 2.5], xmin=0, xmax=len(labels), ls='dotted', alpha=1)
    #     # ax[count].fill_between(x=[0, len(labels)], y1=-1.5, y2=-.5, alpha=0.2, color='r')
    #     # ax[count].fill_between(x=[0, len(labels)], y1=-.5, y2=.5, alpha=0.2, color='g')
    #     # ax[count].fill_between(x=[0, len(labels)], y1=.5, y2=1.5, alpha=0.2, color='y')
    #     for timings in timing_slices:
    #         ax[count].fill_between(x=[timings.seconds.iloc[0], timings.seconds.iloc[-1]], y1=0, y2=1, alpha=.5, color='g')

    # plt.show()

    import pdb
    pdb.set_trace()


    # # plot_labeled_paths(centroids, datasets)
    # # get slices for ruck periods
    # ruck_slices, cent_slices = get_break_times(datasets, centroids, rest=False)
    # # re-orient these ruck periods
    # for cents, rucks in zip(cent_slices, ruck_slices):
    #     # if 'PLT1SQ2' in rucks[0].attrs['name']:
    #     # Orient ruck periods
    #     ruck_slices_oriented, forward_angles_ruck = orient_geom(rucks, cents)
    #     # norm_rest, _ = normalize_rest(rucks, cents)
    #     # # Extract features from oriented ruck periods
    #     # features = features_after_orientation(ruck_slices_oriented)
    #     # prep dfs for plotting
    #     ruck_oriented_prepped = prep_df(ruck_slices_oriented, decimate=10)
    #     # ruck_oriented_prepped = prep_df(norm_rest)
    #     # plot these dfs
    #     import pdb
    #     pdb.set_trace()
    #     joint_subplots(ruck_oriented_prepped, rucks[0].attrs['name'], rucks, rest=False)
    #     # make gifs
    #     # make_gifs(ruck_oriented_prepped)