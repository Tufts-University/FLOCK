"""
Functions for extracting spatial features
"""


from scipy.spatial import Voronoi, ConvexHull
from tqdm import tqdm
from shapely.geometry import MultiPoint, Point, Polygon
import numpy as np
import pandas as pd





def get_cent_dist(rucks, UTM=True):
    """
    Get each soldier distance to the group's centroid
    First step in calculating 'Stretch Index', can also use for individual's 'stretch'

    Args:
        rucks (list): list of movement period DataFrames
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.

    Returns:
        cent_dists (list): list of DataFrames for each movement period, with centroid-distance time-series for each soldier
    """

    # intialize list
    cent_dists = []

    # loop through movement periods (rucks)
    for ruck in tqdm(rucks):
        
        # choose units
        if UTM:
            X = 'UTM_x'
            Y = 'UTM_y'
        else:
            X = 'longitude'
            Y = 'latitude'

        # get centroid (mean)
        cent = pd.concat([ruck[X].mean(axis=1), ruck[Y].mean(axis=1)], axis=1, keys=[X, Y])

        cent_dist = []

        # loop through soldiers
        for name in ruck['longitude'].columns:

            # get one soldiers's data
            this_soldier = pd.concat([ruck[X,name], ruck[Y,name]], axis=1)

            # Get diff from cent
            this_soldier_dists = this_soldier-cent.values

            # euclidean distance
            this_soldier_dist = np.sqrt(this_soldier_dists[X]**2  + this_soldier_dists[Y]**2)

            # append for this dataset
            cent_dist.append(this_soldier_dist)
        
        # create df with columns as indiv names
        cent_dist_df = pd.concat(cent_dist, axis=1)

        # rename df
        cent_dist_df.attrs['name'] = ruck.attrs['name']

        # append to list of distances
        cent_dists.append(cent_dist_df)
    
    
    return cent_dists





def neighbor_dists(rucks, UTM=True):
    """
    Get distance to nearest neighbor for each soldier

    Args:
        rucks (list): list of movement period DataFrames
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.

    Returns:
        dists (list): list of DataFrames for each movement period, with nearest neighbor distance time series for each soldier
    """

    # initialize list of dists
    dists = []
    
    # Choose units 
    if UTM:
        X = 'UTM_x'
        Y = 'UTM_y'
    else:
        X = 'longitude'
        Y = 'latitude'
    
    # loop through movement periods
    for ruck in rucks:

        # init list for this ruck
        ruck_nns = []
        
        # loop thorugh names
        for name in ruck['longitude'].columns:
            
            # init list for this soldier
            this_soldier_neighbors = []
            
            # get this soldiers locations
            this_soldier = pd.concat([ruck[X, name], ruck[Y, name]], axis=1)
            
            # get other soldier locations
            other_soldiers = [pd.concat([ruck[X, oth_name], ruck[Y, oth_name]], axis=1) for oth_name in ruck['longitude'].columns if not oth_name == name]
            
            # loop through soldiers
            for oth_sold in other_soldiers:
                
                # get x and y distances
                XYdistances = this_soldier - oth_sold.values
                
                # get euclidean distances
                distances = np.sqrt(XYdistances[X]**2 + XYdistances[Y]**2)
                
                # append the distances to neighbors
                this_soldier_neighbors.append(distances)
            
            # concat all neighbor distances for this soldier
            all_neighbors = pd.concat(this_soldier_neighbors, axis=1)
            
            # get distance of nearest neighbor over time for this soldier
            nearest_neighbors = pd.Series(all_neighbors.min(axis=1), name=name)
            
            # append this soldiers' nns to list
            ruck_nns.append(nearest_neighbors)
        
        # create df of all soldiers nns over time for this ruck
        all_nearest_neighbors = pd.concat(ruck_nns, axis=1)
        
        # append to final list
        dists.append(all_nearest_neighbors)

    return dists





def get_surface_area(rucks, UTM=True):
    """
    Find the convex hull and calculate the surface area 
    Creates a time series of surface areas for that squad at each timepoint

    Args:
        rucks (list): list of movement period DataFrames
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.

    Returns:
        surface_areas (list): list of surface area time-series for movement periods
    """

    # init list
    surface_areas = []

    # loop through movement periods (rucks)
    for ruck in tqdm(rucks):
        
        # Chose units 
        if UTM:
            X = 'UTM_x'
            Y = 'UTM_y'
        else:
            X = 'longitude'
            Y = 'latitude'
        
        # get surface area 
        SAs = []

        # ruck.dropna(inplace=True)

        # interate through timepoints
        for _ , row in ruck.iterrows():
            if row.isna().any():
                SAs.append(np.nan)
            else:
                points = [[row[X,S], row[Y,S]] for S in row[X].index]
                Hull = ConvexHull(points)
                SAs.append(Hull.area)
        # make df of surface areas across time
        SA_df = pd.DataFrame(SAs, columns=[ruck.attrs['name']], index=ruck.index)
    
        surface_areas.append(SA_df)
    
    return surface_areas



# using voronoi_finite_polygons_2d function from https://stackoverflow.com/questions/34968838/python-finite-boundary-voronoi-cells
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        try: 
            ridges = all_ridges[p1]
        except KeyError: 
            raise KeyError("p1 not found in all_ridges. This usually means that there were duplicates in your input points. Try dropping duplicates and NaN's from the data you put into Voronoi().")
            
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)




def get_voronoi_areas(rucks, UTM=True):
    """
    Get the area for each soldier' voronoi space inside of the group's convex hull
    Both the raw area and the ration of individual area / total area

    Args:
        rucks (list): list of movement period DataFrames
        UTM (bool, optional): True if using UTM data, false if GPS data. Defaults to True.

    Raises:
        ValueError: Voronoi calculation Requires 2D input
        KeyError: Voronoi calculation can't handle NaN values

    Returns:
        voronoi_areas (list): list of voronoi area time series for each movement period
        voronoi_ratios (list): list of voronoi area ratio time series for each movement period
    """


    # initialise lists
    voronoi_areas = []
    voronoi_ratios = []

    # loop through movement periods (rucks)
    for ruck in tqdm(rucks):
        
        # Chose units 
        if UTM:
            X = 'UTM_x'
            Y = 'UTM_y'
        else:
            X = 'longitude'
            Y = 'latitude'
        
        
        # get Voronoi area and ratio
        VAs = []
        VRs = []

        # ruck.dropna(inplace=True)

        for ix , row in ruck.iterrows():
            # if any nan make all nan
            if row.isna().any():
                VAs.append([np.nan]*len(ruck.latitude.columns))
                VRs.append([np.nan]*len(ruck.latitude.columns))
            else:
                points = [[row[X,S]-ruck[X].mean().mean(), row[Y,S]-ruck[Y].mean().mean()] for S in row[X].index]

                points = np.array(points)

                # Calculate Voronoi diagram
                vor = Voronoi(points)

                # get regions and vertices
                regions, vertices = voronoi_finite_polygons_2d(vor)

                # Create 'points' object
                pts = MultiPoint([Point(i) for i in points])

                # create convex hull mask
                mask = pts.convex_hull

                # intialisze areas and ratios
                areas = []
                ratios = []

                # loop through regions
                for region in regions:

                    # get polygon
                    polygon = vertices[region]

                    # get shape of poly
                    shape = list(polygon.shape)
                    shape[0] += 1

                    # find polygon intersections
                    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)

                    # if polygon is empty make size 0
                    if p.is_empty:
                        areas.append(0)
                        ratios.append(0)
                    else:
                        # calculate area of voronoi polygon within convex hull
                        poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
                        area = MultiPoint([Point(i) for i in poly]).convex_hull.area
                        # append area value
                        areas.append(area)
                        # append ration value
                        ratios.append(area/mask.area)

                # append area and ratios
                VAs.append(areas)
                VRs.append(ratios)

        # Create dfs to return
        VA_df = pd.DataFrame(VAs, columns=ruck.latitude.columns, index=ruck.index) 
        voronoi_areas.append(VA_df)
        VR_df = pd.DataFrame(VRs, columns=ruck.latitude.columns, index=ruck.index) 
        voronoi_ratios.append(VR_df)


    return voronoi_areas, voronoi_ratios


