'''
Functions for using directional correlation time delay metrics as leadership metrics

'''

import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os



def get_directional_corr(movement_periods, names, UTM=True, threshold = 10, window_length = 9):
    """
    Get the Directional Correlation of soldiers
    For finding the Directional Correlation time-delay as a leasership metric
    and the ratio of time spent 'highly correlated' as a sychronicity metric (Highly Correlated Segments (HCS))

    Finding the normalized velocity vectors and taking the dot product between pairs as 'correlation' for different time delays -4 to 4 seconds
    Time delay of maximum correlation is the directional ingfluence time delay measure 

    Returns time delay dfs for each soldier in each movement period, HCS ratios for each soldier in each period, and a graph representation of leadership heirarchy for each movement period


    Args:
        movement_periods (list): list of DataFrames for each movement period that have been further smoothed
        names (list): list of names from this squad
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.
        threshold (int, optional): Distance threshold, Directional Correlation only calculated if within this threshold (meters) for window_length of time (seconds). Defaults to 10.
        window_length (int, optional): Duration (in seconds) that two soldiers must be in proximity (below threshold distance) in order to calculate directional correlation. Defaults to 9.

    Returns:
        time_delay_dfs (list): list of correaltional time delays over time as dfs for each soldier during each movement period
        HCS_ratio_dfs (list): list Series for each movement period, with an HSC ratio for each player
        graphs (list): list of networkx directed graphs representing leadership heirarchy, edges pointing from leader to follower
    """

    # initialize vector functions
    def magnitude(v):
        return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))
    def dot(u, v):
        return sum(u[i]*v[i] for i in range(len(u)))
    def normalize(v):
        vmag = magnitude(v)
        return [ v[i]/vmag  for i in range(len(v)) ]
    
    # get prefered units
    if UTM:
        X = 'UTM_x'
        Y = 'UTM_y'
    else:
        X = 'longitude'
        Y = 'latitude'

    # initialize the time delay (TD) df
    time_delay_dfs = []
    # init Highly Correlated Segments (HCS) ratios
    HCS_ratio_dfs = []
    # init list of grapph representations
    graphs = []
    
    # find pairs of soldiers
    name_pairs = list(combinations(names, 2))
    
    # loop through movement_periods
    for ruck in tqdm(movement_periods):
        
        # initialize directed graph (this ruck)
        G = nx.DiGraph()
        for n in names: G.add_node(n)

        # initialize list of TD series'
        time_delay_ser = []
        HCS_ratio_ser = []

        
        # get normalized 'flock' velocity
        flock_vs = pd.concat((ruck[X].mean(axis=1),ruck[Y].mean(axis=1)), axis=1).diff()
        flock_normVs = pd.DataFrame([normalize(v) for v in flock_vs.to_numpy()])

        # loop through soldiers
        for name_pair in name_pairs:

            # get pair names
            this_name = name_pair[0]
            other_name = name_pair[1]

            # get this soldiers's data
            this_soldier = pd.concat([ruck[X,this_name], ruck[Y,this_name]], axis=1).dropna()

            # get other soldiers's data
            other_soldier = pd.concat([ruck[X,other_name], ruck[Y,other_name]], axis=1).dropna()

            # get this soldier's normalized velocity vectors
            this_soldier_normVs = pd.DataFrame([normalize(v) for v in this_soldier.diff().to_numpy()], columns=this_soldier.columns)
            
            # get other soldier's normalized velocity vectors
            other_soldier_normVs = pd.DataFrame([normalize(v) for v in other_soldier.diff().to_numpy()], columns=other_soldier.columns)
            
            # get dists between soldiers
            distsXY = this_soldier - other_soldier.values

            # get projected distance onto the direction of motion of the whole flock
            dists = pd.Series([dot(u,v) for u,v in zip(distsXY.to_numpy(), flock_normVs.to_numpy())])
            dists.name = 'dists'
            
            # get where dists <= threshold
            dists_bool = dists.abs() <= threshold
            this_comp = pd.concat([this_soldier_normVs, other_soldier_normVs, dists_bool], axis=1)

            # initialize list of TDs 
            time_delays = []
            dot_prods = []
            time_delay_list = []
            dot_prod_list = []
            
            # apply time delay calculation with custom rolling window
            for idx in range(len(this_comp)-(window_length-1)):

                # get this 'window_length' second window
                x = this_comp.iloc[idx:idx+ window_length]
                
                # check to be sure this soldier is within [threshold] distance
                if x.dists.all(): 
                    
                    # get the middle vector for the OG soldier
                    this_vector = x[[x.columns[0], x.columns[1]]].iloc[window_length//2].values
                    
                    # get vectors to compare
                    comp_vectors = x[[x.columns[2], x.columns[3]]].values
                    
                    # compare vectors with time delays
                    dot_prod_window = pd.Series([dot(this_vector, v) for v in comp_vectors], index=np.arange(-(window_length//2),(-(window_length//2) + window_length)))
                    
                    # find time delay with largest dot product (now after corr_thresh)
                    time_delays.append(dot_prod_window.idxmax())
                    dot_prods.append(dot_prod_window.max())

                else:
                    # if not within distance for full 9 seconds, append nan
                    time_delays.append(np.nan)
                    dot_prods.append(np.nan)
                
            # corr threshold
            corr_thresh = 0.99
            
            # add an directed edge (leader to follower) for the pair of soldiers
            # if the correlation is above corr_thresh
            average_HCS_TD = pd.Series(time_delays)[pd.Series(dot_prods) > corr_thresh].mean()

            # if positive, this_name to other_name
            if   average_HCS_TD > 0:
                G.add_edge(this_name, other_name, weight = average_HCS_TD)
                # print('adding edge: ' +this_name+' '+other_name+' '+str(average_HCS_TD))

            # if negative, other_name to this_name
            elif average_HCS_TD < 0:
                G.add_edge(other_name, this_name, weight = -average_HCS_TD)
                # print('adding edge: ' +other_name+' '+this_name+' '+str(-average_HCS_TD))

            # append all of the time delays for this soldier (when above corr_thresh)
            # append all of the time delays (-) for other soldier (when above corr_thresh)
            time_delay_list.append( pd.Series(time_delays, name = this_name)[pd.Series(dot_prods) > corr_thresh])
            time_delay_list.append(-pd.Series(time_delays, name = other_name)[pd.Series(dot_prods) > corr_thresh])
            max_dots = pd.Series(dot_prods, name = this_name)

            # calculate HCS ratio for this pair
            if max_dots.isna().all():
                dot_prod_list.append(np.nan)
            else:
                HCS_interval = 6
                HCS_threshold = 0.99
                # find values that are not nan (in proximi)
                prox_time = max_dots.rolling(HCS_interval).mean().dropna().shape[0]
                H_time = (max_dots.rolling(HCS_interval).mean() > HCS_threshold).sum()
                assert not prox_time == np.nan, 'nan value in HCS (p)'
                assert not H_time == np.nan, 'nan value in HCS (h)'
                # assert not prox_time == 0, '0 in HCS (p)'
                # assert not H_time == 0, '0 in HCS (h)'
                if H_time == 0 or prox_time == 0:
                    dot_prod_list.append(0)
                dot_prod_list.append(H_time/prox_time)
                
            # concat all comparison soldiers TDs and add to list
            time_delay_ser.append(pd.concat(time_delay_list, axis=1).reset_index(drop=True))
            HCS_ratio_ser.append(pd.Series(dot_prod_list, name = this_name))
            HCS_ratio_ser.append(pd.Series(dot_prod_list, name = other_name))

        # create df of all soldiers TDs with all other soldiers
        time_delay_dfs.append(pd.concat([pd.Series(pd.concat(time_delay_ser, axis=1)[name].stack().values, name=name) for name in names], axis=1))
        # average HCS across soldier comparisons each movement period
        HCS_ratio_dfs.append(pd.concat(HCS_ratio_ser, axis=1).groupby(level=0, axis=1).mean())
        # append graphs
        graphs.append(G)

    return time_delay_dfs, HCS_ratio_dfs, graphs






def leadership_graph_ani(time_delay_dfs, graphs, names, sq_name, show=False):
    """
    Plot an animation of leadership graphs for each movememt period
    One frame is one movement period leadership heirarchy
    
    The leadership features have been extracted in get_directional_corr()

    Args:
        time_delay_dfs (list): list of correaltional time delays over time as dfs for each soldier during each movement period
        graphs (list): list of networkx directed graphs representing leadership heirarchy, edges pointing from leader to follower
        names (list): list of names from this squad as list of str
        sq_name (str): name of squad as str
        show (bool, optional): if the plot should be displayed, saved if false. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    # Build plot
    fig, ax = plt.subplots(figsize=(10,10))

    # initialize frame count
    n=0

    # create frame-update function
    def update(n):

        # clear the plot
        ax.clear()

        # get graph of [n] movenent periods
        G = graphs[n]      
        tds = time_delay_dfs[n].mean()
        # tds = pd.concat(time_delay_dfs).mean()

        # get Q-heir metric
        Greats = 0
        Lesses = 0
        for e in G.edges: 
            if tds[e[0]] > tds[e[1]]:
                Greats+=1
                # print(e[0]+' Greater than '+e[1])
            elif tds[e[0]] < tds[e[1]]:
                Lesses+=1
                # print(e[0]+' Less than '+e[1])
        Q_heir = Greats/(Greats+Lesses)
        # print('Q_heir is %.2f '%Q_heir)
        # print('# of cycles: %.2f '%len(sorted(nx.simple_cycles(G))))

        options = {
            'node_size': 3000,
            'width': 3,
            'arrowsize': 30,
            'ax':ax,
        }

        pos = nx.spring_layout(G)
        wpos = dict([(p,np.array([count/len(names),w]))for count, ((p,q), (i,w)) in enumerate(zip(pos.items(), tds.items())) if p==i])
        attrs = {'value': dict(tds)}
        nx.set_node_attributes(G, values=attrs)
        nx.draw_networkx(G, wpos, **options)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = [{k: round(v, 2) for k, v in edge_labels.items()}][0]
        nx.draw_networkx_edge_labels(G, wpos, edge_labels)

        mins = time_delay_dfs[n].shape[0]//60 
        # # Scale plot ax
        ax.set_title("Movement period %d  "%n +  ": %d minutes long"%mins+'\nQ_heir: %.2f '%Q_heir+' # of loops: %.2f '%len(sorted(nx.simple_cycles(G))) , fontweight="bold")
        ax.set_xlabel('Group Members')
        ax.set_ylabel('Directional Correlation\nTIme Delay\n(Leadership score)')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_yticks(tds)
        ax.set_yticklabels(round(tds, 2))
        ax.set_xticks(np.arange(0,1,1/len(pos.keys() )) )
        ax.set_xticklabels(pos.keys() )
        n+=1

        plt.axis('on')

    ani = FuncAnimation(fig, update, frames=len(graphs), interval=1000, repeat=True)
    if show:
        plt.show()
    else:
        ani.save(os.getcwd() + r'\Figures\Pairwise_Leadership_Animation_'+sq_name+'.gif')

    return None




def dir_corr_graph_comparison(graphs):
    """
    Get metrics for the consistency of the leadership heirarchy graph over movement periods
    get the edit distance between movement period graphs
    also compare adjacency martrices across movement period graphs

    Args:
        graphs (list): list of networkx directed graphs representing leadership heirarchies in different movement periods

    Returns:
        G_consist (float): average edit diatance over all pairs of movement periods
        G_adj_consist (DataFrame): adjacency matrix differences for each pair of movement periods
    """

    # get pairs of graphs (campare all movement periods with eachother)
    G_ps = [(a, b) for idx, a in enumerate(graphs) for b in graphs[idx + 1:]]

    # init lists of differences
    edit_dists = []
    adj_diffs = []

    # loop through pairs
    for Gs in G_ps:

        # get edit distance
        # normalize by getting overall number of edges
        # now edit distance is a ratio of edges to be changed vs total edges
        edit_dists.append(nx.graph_edit_distance(Gs[0], Gs[1]) / np.mean([Gs[0].size(), Gs[1].size()]))

        # get adjacency matrix for manual comparison
        Gr_1 = nx.to_pandas_adjacency(Gs[0])
        # make directed edges negative in opposite direction
        G1complete = Gr_1-Gr_1.T
        # get adjacency matrix for manual comparison
        Gr_2 = nx.to_pandas_adjacency(Gs[1])
        # make directed edges negative in opposite direction
        G2complete = Gr_2-Gr_2.T
        # append adjacency matrix comparison (euclidean distance)
        adj_diffs.append(np.sqrt(((G1complete-G2complete)**2).mean()))

    G_consist = np.mean(edit_dists)
    G_adj_consist = pd.concat(adj_diffs, axis=1).T

    return G_consist, G_adj_consist




if __name__ == '__main__':
    '''
    For testing above functions with command line 

    '''