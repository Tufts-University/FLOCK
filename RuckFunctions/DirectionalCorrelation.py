'''
Functions for using the directional correlation metric for extracting leadership metrics

'''

# from DataLoading import *
# from PACS import *
# from FeatureExtraction import *
# from Clustering import *
# from Preprocessing import *

import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def directional_corr(rucks, names=None, UTM=True, threshold = 10, window_length = 9):
    '''
    get the directional correlation for the soldiers (pairwise)
    
    find the normalized velocity vector, get the dot product (using different time lags)
    find the most correlated time lag

    '''
    assert not names==None, 'Input names to Directional_corr'

    # initialize vector functions
    def magnitude(v):
        return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))
    def dot(u, v):
        return sum(u[i]*v[i] for i in range(len(u)))
    def normalize(v):
        vmag = magnitude(v)
        return [ v[i]/vmag  for i in range(len(v)) ]
    
    if UTM:
        X = 'UTM_x'
        Y = 'UTM_y'
    else:
        X = 'longitude'
        Y = 'latitude'

    # initialize the time delay (TD) df
    time_delay_dfs = []
    HCS_ratio_dfs = []
    graphs = []
    
    # loop through rucks
    for ruck in tqdm(rucks):

        
        # initialize directed graph (this ruck)
        G = nx.DiGraph()
        for n in names: G.add_node(n)

        # initialize list of TD series'
        time_delay_ser = []
        HCS_ratio_ser = []

        # loop through soldiers
        for name in names:

            # get one soldiers's data
            this_soldier = pd.concat([ruck[X,name], ruck[Y,name]], axis=1).dropna()

            # get other soldier's data
            other_soldiers = [pd.concat([ruck[X,othername], ruck[Y,othername]], axis=1).dropna() for othername in names if not othername==name]
            
            # get this soldiers normalized velocity vectors
            this_soldier_normVs = pd.DataFrame([normalize(v) for v in this_soldier.diff().to_numpy()], columns=this_soldier.columns)
            
            # get other soldiers normalized velocity vectors
            other_soldiers_normVs = [pd.DataFrame([normalize(v) for v in s.diff().to_numpy()],columns=s.columns) for s in other_soldiers]

            # get normalized 'flock' velocity
            flock_vs = pd.concat((ruck[X].mean(axis=1),ruck[Y].mean(axis=1)), axis=1).diff()
            flock_normVs = pd.DataFrame([normalize(v) for v in flock_vs.to_numpy()])
            
            # initialize list of TDs for all other soldiers
            time_delay_list = []
            dot_prod_list = []
            
            # loop thorugh other soldier's normVs and get a value for each window
            # except windows where the distance is greater than [threshold]
            for other_soldier, other_soldier_normVs in zip(other_soldiers, other_soldiers_normVs):
                
                # get the name of the soldier to compare with this_soldier
                other_name = other_soldier.columns[0][1]

                # get dists between soldiers
                distsXY = this_soldier - other_soldier.values

                # get projected distance onto the direction of motion of the whole flock
                dists = pd.Series([dot(u,v) for u,v in zip(distsXY.to_numpy(), flock_normVs.to_numpy())])
                dists.name = 'dists'

                # get where dists <= threshold
                dists_bool = dists.abs() <= threshold
                this_comp = pd.concat([this_soldier_normVs, other_soldier_normVs, dists_bool], axis=1)

                # init list of time delays / dot products
                time_delays = []
                dot_prods = []
                
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

                        # now done after getting values all movement period
                        # # corr threshold
                        # corr_thresh = 0.99
                        # # if dot_prod max is above corr_thresh, add edge
                        # if dot_prod_window.max() >= corr_thresh:
                        #     # find time delay with largest dot product
                        #     time_delays.append(dot_prod_window.idxmax())
                        #     dot_prods.append(dot_prod_window.max())
                            # # add edge from leader to follower (for using multi-edge)
                            # if dot_prod_window.idxmax() >= 0:
                            #     G.add_edge(name, other_name, weight = dot_prod_window.idxmax())
                            #     # print('adding edge: ' +name+' '+other_name+' '+str(average_TD))
                            # else:
                            #     G.add_edge(other_name, name, weight = -dot_prod_window.idxmax())
                            #     # print('adding edge: ' +other_name+' '+name+' '+str(-average_TD))
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
                    G.add_edge(name, other_name, weight = average_HCS_TD)
                    # print('adding edge: ' +name+' '+other_name+' '+str(average_TD))

                # if negative, other_name to this_name
                elif average_HCS_TD < 0:
                    G.add_edge(other_name, name, weight = -average_HCS_TD)
                    # print('adding edge: ' +other_name+' '+name+' '+str(-average_TD))

                # append all of the time delays with this soldier (when above corr_thresh)
                time_delay_list.append(pd.Series(time_delays, name = name)[pd.Series(dot_prods) > corr_thresh])
                max_dots = pd.Series(dot_prods, name = name)

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
            time_delay_ser.append(pd.concat(time_delay_list).reset_index(drop=True))
            HCS_ratio_ser.append(pd.Series(dot_prod_list, name = name))

        # create df of all soldiers TDs with all other soldiers
        time_delay_dfs.append(pd.concat(time_delay_ser, axis=1))
        HCS_ratio_dfs.append(pd.concat(HCS_ratio_ser, axis=1))
        graphs.append(G)

    return time_delay_dfs, HCS_ratio_dfs, graphs






def new_pairs_directional_corr(rucks, names, UTM=True, threshold = 10, window_length = 9):
    """
    Get the Directional Correlation of soldiers
    For finding the Directional Correlation time-delay as a leasership metric
    and the ratio of time spent 'highly correlated' as a sychronicity metric

    Finding the normalized velocity vectors and taking the dot product as 'correlation'


    Args:
        rucks (list): list of DataFrames for each movement period that have been further smoothed
        names (list): list of names from this squad
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.
        threshold (int, optional): Distance threshold, Directional Correlation only calculated if within this threshold (meters) for window_length of time (seconds). Defaults to 10.
        window_length (int, optional): Duration (in seconds) that two soldiers must be in proximity (below threshold distance) in order to calculate directional correlation. Defaults to 9.

    Returns:
        time_delay_dfs (list): list of correaltional time delays ovedr time as dfs for each soldier during each mopvement period
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
    
    # loop through rucks
    for ruck in tqdm(rucks):
        
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
    '''
    Plot an animation of leadership graphs for each movememt period
    
    The leadership features have been extracted in directional_corr()
    '''

    Squad_leaders = ['MPA088', 'MPA098', 'MPA090', 'MPA110', 'MPA100', 'MPA107', 'MPA132', 'MPA149', 'MPA161', ]
    A_leaders =     ['MPA086', 'MPA094', 'MPA096', 'MPA113', 'MPA126', 'MPA118', 'MPA133', 'MPA146', 'MPA147', ]
    B_leaders =     ['MPA092', 'MPA091', 'MPA085', 'MPA121', 'MPA111', 'MPA117', 'MPA130', 'MPA140', 'MPA168', ] 
    
    # Build plot
    fig, ax = plt.subplots(figsize=(10,10))

    n=0

    def update(n):
        ax.clear()

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

        colors = []
        for node in G:
            if node in A_leaders or node in B_leaders: colors.append('blue'),
            elif node in Squad_leaders: colors.append('green'),
            else: colors.append('grey')

        options = {
            'node_color': colors,
            'node_size': 3000,
            'width': 3,
            # 'arrowstyle': '-|>',
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


        secs = time_delay_dfs[n].shape[0]//60 
        # # Scale plot ax
        ax.set_title("Movement period %d  "%n +  ": %d minutes long"%secs+'\nBlue : Squad Leader, Green: Team Leaders (A&B)\nQ_heir: %.2f '%Q_heir+' # of loops: %.2f '%len(sorted(nx.simple_cycles(G))) , fontweight="bold")
        ax.set_xlabel('Soldiers')
        ax.set_ylabel('Directional Correlation\nTIme Delay\n(Leadership score)')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_yticks(tds, round(tds, 2))
        ax.set_xticks(np.arange(0,1,1/len(pos.keys() )), pos.keys() )
        n+=1

        plt.axis('on')

    ani = FuncAnimation(fig, update, frames=len(graphs), interval=1000, repeat=True)
    if show:
        plt.show()
    else:
        ani.save(os.getcwd() + r'\..\Figures\Pairwise_Leadership_Animation'+sq_name+'.gif')

    return ani




def pairs_directional_corr(rucks, names=None, UTM=True, dist_threshold = 10, window_length = 9):
    '''
    get the directional correlation for the soldiers (pairwise)
    
    find the normalized velocity vector, get the dot product (using different time lags)
    find the most correlated time lag

    '''
    assert not names==None, 'Input names to pairs_directional_corr'

    # initialize vector functions
    def magnitude(v):
        return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))
    def dot(u, v):
        return sum(u[i]*v[i] for i in range(len(u)))
    def normalize(v):
        vmag = magnitude(v)
        return [ v[i]/vmag  for i in range(len(v)) ]
    
    if UTM:
        X = 'UTM_x'
        Y = 'UTM_y'
    else:
        X = 'longitude'
        Y = 'latitude'

    # initialize the time delay (TD) df
    time_delay_dfs = []
    dot_prod_dfs = []
    graphs = []

    # find pairs
    name_pairs = list(combinations(names, 2))
    
    # loop through rucks
    for ruck in tqdm(rucks):

        
        # initialize directed graph (this ruck)
        G = nx.DiGraph()
        for n in names: G.add_node(n)

        # initialize list of TD series'
        time_delay_ser = []
        dot_prod_ser = []

        # get normalized 'flock' velocity
        flock_vs = pd.concat((ruck[X].mean(axis=1),ruck[Y].mean(axis=1)), axis=1).diff()
        flock_normVs = pd.DataFrame([normalize(v) for v in flock_vs.to_numpy()])

        # loop through soldiers
        for name_pair in name_pairs:

            
            # initialize list of TDs for all other soldiers
            time_delay_list = []
            dot_prod_list = []

            # get these soldier's data
            A_name = name_pair[0]
            B_name = name_pair[1]
            soldier_A = pd.concat([ruck[X,A_name], ruck[Y,A_name]], axis=1).dropna()
            soldier_B = pd.concat([ruck[X,B_name], ruck[Y,B_name]], axis=1).dropna()
            # get their normalized vectors
            A_normVs = pd.DataFrame([normalize(v) for v in soldier_A.diff().to_numpy()], columns=soldier_A.columns)
            B_normVs = pd.DataFrame([normalize(v) for v in soldier_B.diff().to_numpy()], columns=soldier_B.columns)
            # get their distance velcors
            soldier_dists = (soldier_A - soldier_B.values)
            # get projected distance onto the direction of motion of the whole flock
            dists = pd.Series([dot(u,v) for u,v in zip(soldier_dists.to_numpy(), flock_normVs.to_numpy())])
            dists.name = 'dists'
            # get where dists <= dist_threshold
            dists_bool = dists.abs() <= dist_threshold
            # put comparison df together
            AB_comp = pd.concat([A_normVs, B_normVs, dists_bool], axis=1)
            BA_comp = pd.concat([B_normVs, A_normVs, dists_bool], axis=1)
            comp_df = pd.concat([A_normVs, B_normVs, dists_bool], axis=1)



            # init list of time delays / dot products
            AB_time_delays = []
            AB_dot_prods = []
            BA_time_delays = []
            BA_dot_prods = []

            
            time_delays = []
            dot_prods = []

            edge12 = []
            edge21 = []
            
            # apply time delay calculation with custom rolling window
            for idx in range(len(comp_df)-(window_length-1)):

                # get this 'window_length' second window
                x = comp_df.iloc[idx:idx+ window_length]
                
                # check to be sure this soldier is within [threshold] distance
                if x.dists.all():
                    
                    # get the middle vector for the OG soldier
                    vector_1 = x[[x.columns[0], x.columns[1]]].iloc[window_length//2].values
                    vector_2 = x[[x.columns[2], x.columns[3]]].iloc[window_length//2].values
                    
                    # get vectors to compare
                    set_1 = x[[x.columns[0], x.columns[1]]].values
                    set_2 = x[[x.columns[2], x.columns[3]]].values
                    
                    # compare vectors with time delays
                    dot_prod_window_1 = pd.Series([dot(vector_1, v) for v in set_2], index=np.arange(-(window_length//2),(-(window_length//2) + window_length)))
                    dot_prod_window_2 = pd.Series([dot(vector_2, v) for v in set_1], index=np.arange(-(window_length//2),(-(window_length//2) + window_length)))
                    
                    if   dot_prod_window_1.idxmax() > 0:
                        edge12.append(dot_prod_window_1.idxmax())
                    elif dot_prod_window_1.idxmax() < 0:
                        edge21.append(-dot_prod_window_1.idxmax())
                    print(dot_prod_window_1.idxmax(), dot_prod_window_2.idxmax())
                    if dot_prod_window_2.idxmax()==0:
                        break

                    # find time delay with largest dot product
                    time_delays.append(dot_prod_window.idxmax())
                    dot_prods.append(dot_prod_window.max())

                else:
                    # if not within distance, append nan
                    time_delays.append(np.nan)
                    dot_prods.append(np.nan)



            # apply time delay calculation with custom rolling window
            for idx in range(len(AB_comp)-(window_length-1)):

                # get this 'window_length' second window
                x = AB_comp.iloc[idx:idx+ window_length]
                
                # check to be sure this soldier is within [threshold] distance
                if x.dists.all():
                    
                    # get the middle vector for the OG soldier
                    this_vector = x[[x.columns[0], x.columns[1]]].iloc[window_length//2].values
                    
                    # get vectors to compare
                    comp_vectors = x[[x.columns[2], x.columns[3]]].values
                    
                    # compare vectors with time delays
                    dot_prod_window = pd.Series([dot(this_vector, v) for v in comp_vectors], index=np.arange(-(window_length//2),(-(window_length//2) + window_length)))
                    
                    # find time delay with largest dot product
                    AB_time_delays.append(dot_prod_window.idxmax())
                    AB_dot_prods.append(dot_prod_window.max())

                else:
                    # if not within distance, append nan
                    AB_time_delays.append(np.nan)
                    AB_dot_prods.append(np.nan)


            # apply time delay calculation with custom rolling window
            for idx in range(len(BA_comp)-(window_length-1)):

                # get this 'window_length' second window
                x = BA_comp.iloc[idx:idx+ window_length]
                
                # check to be sure this soldier is within [threshold] distance
                if x.dists.all():
                    
                    # get the middle vector for the OG soldier
                    this_vector = x[[x.columns[0], x.columns[1]]].iloc[window_length//2].values
                    
                    # get vectors to compare
                    comp_vectors = x[[x.columns[2], x.columns[3]]].values
                    
                    # compare vectors with time delays
                    dot_prod_window = pd.Series([dot(this_vector, v) for v in comp_vectors], index=np.arange(-(window_length//2),(-(window_length//2) + window_length)))
                    
                    # find time delay with largest dot product
                    BA_time_delays.append(dot_prod_window.idxmax())
                    BA_dot_prods.append(dot_prod_window.max())

                else:
                    # if not within distance, append nan
                    BA_time_delays.append(np.nan)
                    BA_dot_prods.append(np.nan)
            
            # corr threshold
            corr_thresh = 0.99

            BA_mean = pd.Series(BA_time_delays)[pd.Series(BA_dot_prods) > corr_thresh].mean()
            AB_mean = pd.Series(AB_time_delays)[pd.Series(BA_dot_prods) > corr_thresh].mean()

            # get comparison from positive valued time delay
            if BA_mean > AB_mean:
                time_delays = BA_time_delays
                dot_prods = BA_dot_prods
                if BA_mean > 0:
                    G.add_edge(B_name, A_name, weight = BA_mean)
                else:
                    G.add_edge(A_name, B_name, weight = -BA_mean)
            else:
                time_delays = AB_time_delays
                dot_prods = AB_dot_prods
                if AB_mean > 0:
                    G.add_edge(A_name, B_name, weight = AB_mean)
                else:
                    G.add_edge(B_name, A_name, weight = -AB_mean)

        #     # append the time delays with this soldier 
        #     time_delay_list.append(pd.Series(time_delays, name = name))
        #     max_dots = pd.Series(dot_prods, name = name)
        #     # calculate HCS ratio for this pair
        #     if max_dots.isna().all():
        #         dot_prod_list.append(np.nan)
        #     else:
        #         HCS_interval = 6
        #         HCS_threshold = 0.99
        #         # find values that are not nan (in proximi)
        #         prox_time = max_dots.rolling(HCS_interval).mean().dropna().shape[0]
        #         H_time = (max_dots.rolling(HCS_interval).mean() > HCS_threshold).sum()
        #         assert not prox_time == np.nan, 'nan value in HCS (p)'
        #         assert not H_time == np.nan, 'nan value in HCS (h)'
        #         # assert not prox_time == 0, '0 in HCS (p)'
        #         # assert not H_time == 0, '0 in HCS (h)'
        #         if H_time == 0 or prox_time == 0:
        #             dot_prod_list.append(0)
        #         dot_prod_list.append(H_time/prox_time)
                
        #     # concat all comparison soldiers TDs and add to list
        #     time_delay_ser.append(pd.concat(time_delay_list))
        #     dot_prod_ser.append(pd.Series(dot_prod_list, name = name))

        # # create df of all soldiers TDs with all other soldiers
        # time_delay_dfs.append(pd.concat(time_delay_ser, axis=1))
        # dot_prod_dfs.append(pd.concat(dot_prod_ser, axis=1))
        graphs.append(G)

    return graphs




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