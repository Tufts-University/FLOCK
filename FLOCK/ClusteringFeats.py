'''
For extracting features from oriented and segmented data

Using different clustering methods and extracting features from each

The dataset are assumed to be preprocessed 

'''
# imports 


from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import math
from tqdm.notebook import tqdm
import numpy as np
import plotly.express as px
from hdbscan import HDBSCAN
import pandas as pd
import io
import PIL
import os




def cluster_for_separation(datasets, UTM=True, method='HDBSCAN', epsilon=10, min_cluster_size = 2):
    """
    CLuster the data for separation metrics, using the density-based clustering method of choice (DBSCAN or HDBSCAN)

    Args:
        datasets (list): list of movement period DataFrames
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.
        method (string ('DBSCAN' or 'HDBSCAN'), optional): clustering method. Defaults to 'HDBSCAN'.
        epsilon (float, optional): epsilon for clustering method, if applicable. This is the threshold distance for clusters to be separated, preventing micro-cluistering Defaults to 10.
        min_cluster_size(int, optional): minimum number of points to be considered a cluster. Clusters with less than this number will be considered outliers. Defaults to 2.

        
    Returns:
        _type_: _description_
        all_membership_probs (list) : list of dataframes with cluster membership probabilities for each soldier, if method = 'HDBSCAN'
        all_labels (list): list of dataframes with cluster labels for each soldier at each timepoint
        all_scores (list): list of series the silouette scores for each timepoint, if multiple clusters are present, otherwise NaN
    """

    # initialize lists
    all_membership_probs = []
    all_labels = []
    all_scores = []

    # loop through datasets
    for data in datasets:

        # get names
        names = []
        for col in data.columns:

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
                    fitted = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True,
                                    cluster_selection_epsilon=epsilon, min_samples=2, )
                    
                    # fit to data
                    fitted.fit(points)

                    # append probabilities (HDBSCAN membership probability)
                    inertias_data.append(fitted.probabilities_)

                # if DBSCAN append nan value to porbabilities
                elif method == 'DBSCAN':
                    
                    # initialize model
                    fitted = DBSCAN(eps = epsilon, min_samples = min_cluster_size, )

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
        all_membership_probs.append(pd.DataFrame(inertias_data, columns=names))
        all_labels.append(pd.DataFrame(labels_data, columns=names))
        all_scores.append(pd.Series(scores_data))

    return all_membership_probs, all_labels, all_scores




def get_outlier_time(all_labels):
    """
    get the amount of time each soldier is an outlier from cluster_for_separation label outputs

    Args:
        all_labels (list of DataFrames): list of clustering label dataframes

    Returns:
        outlier_times (list of Series): amount of time each soldier is considered an outlier (label = -1) for each movement period dataframe
    """

    outlier_times = []

    # loop through movement periods
    for labels in all_labels:

        # if no outlier labels
        if not -1.0 in labels.values:

            # return 0 for all members
            outlier_time = pd.Series([0]*len(labels.columns), index=labels.columns)

            # rename output series
            outlier_time.name = 'number of samples as outlier'

            # append output to final list
            outlier_times.append(outlier_time)

        else:
            # get a count for how many samples are '-1' for each individual
            outlier_time = labels.apply(pd.value_counts).loc[-1].fillna(0)

            # rename output series
            outlier_time.name = 'number of samples as outlier'

            # append output to final list
            outlier_times.append(outlier_time)

    return outlier_times




def prep_cluster_df(datasets, all_labels, change_units=True, decimate=0):
    """
    Prep dataframe for plotting

    Args:
        datasets (list): list of movement period DataFrames
        all_labels (list of DataFrames): list of clustering label dataframes
        change_units (bool, optional): True if units should be changed (change if Long/Lat). Defaults to True.
        decimate (int, optional): Decimation factor for the signal. Defaults to 0.

    Returns:
        prepped_clust_dfs (list of DataFrames): Long form dataframe for seaborn plots (columns=['longitude','latitude','ID','time'])
    """

    # initialize list
    prepped_clust_dfs = []

    print('re-formatting data')

    # loop through dfs and label series
    for df, labels in zip(tqdm(datasets), all_labels):
        
        # decimate the signal (take every X samples) if decimate doesnt = 0 
        if decimate:
            ndf = df[::decimate]
            ndf.attrs['name'] = df.attrs['name']
            df = ndf

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
    """
    make gifs from plot_prepped datasets

    Args:
        prepped_clust_dfs (list of DataFrames): list of dataframes that have been prepped for plotting (long form for seaborn)

    Returns:
        None: None
    """

    # loop through dataframes
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
                color="cluster", title=df.attrs['name']+"_"+str(count),)
                # range_y=[df['Y'].min(), df['Y'].max()], range_x=[df['Y'].min(), df['Y'].max()],#, category_orders = np.arange(len(df.cluster.unique()))) 
                #range_x=[df['X'].min(), df['X'].max()])
                # error_x='X_max', error_x_minus='X_min', error_y='Y_max', error_y_minus='Y_min')
                # error_x='X_err', error_y='Y_err', color_discrete_sequence=sns.color_palette(as_cmap=True))
                # marginal_x='box', marginal_y='box')
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

        def animate(i):
            fig.update(data=fig.frames[i].data)

        
        print('save plot')

        # create animated GIF
        # frames[0].save(
        #         os.getcwd() + '\\Figures\\GIF_' + df.attrs['name']+"_"+str(count)+".gif",
        #         save_all=True,
        #         append_images=frames[1:],
        #         optimize=True,
        #         duration=90,
        #         loop=0            
        #     )
        # print('GIF saved to: ' + os.getcwd() + '\\Figures\\GIF_' + df.attrs['name']+"_"+str(count)+".gif")

        
        # init FuncAnimation
        ani = plt.animation.FuncAnimation(fig, animate, frames=fig.frames, interval=200)

        from IPython.display import HTML
        HTML(ani.to_jshtml())


    return None




if __name__ == '__main__':
    # Initialize 
    None