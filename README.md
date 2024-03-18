<h1 align="center" > FLOCK - Feature Learning of Organized Collective Kinetics </h1>

Welcome to the FLOCK-GPS repository

The purpose of this package is to offer preprocessing and feature extraction methods for assessing group movement dynamics from GPS location data.


## Documentation
Documentation can be found in the Docs folder of this repository


## Quick start



The repository is organized as follows:

## FLOCK functions

* **DataLoading.py**
  * Functions for loading and re-formatting data from a directory of .csv files
  * See the SampleData folder for input data format examples
    
* **Preprocessing.py**
  * Preprocessing functions such as outlier sample detection, interpolation of missing datapoints, path smothing and break detection for identifying movement periods
  * 
    
* **VelocityFeats.py**
  * Feature extraction functions for finding the velocity of each individual over time and features related to difference in velocities across group members
  
* **SpatialFeats.py**
  * Feature extraction functions for spatio-temporal features (spatial features over time). Such as the stretch index, convex hull surface area and voronoi spaces 
  
* **PACS.py**
  * Path-adapted coordinate system transformation
  
* **PacsFeats.py**
  * Feature extraction for path adapted coordinate system tranaformed data. Such as the spatial exploration index of each individual, the nearest neighbor (left/right and front/back), the length/width ratio of the group and the consistency of member positions in different movement periods
  
* **DirectionalCorrelation.py**
  * Directional correlation time delay leadership metrics from Nagy et. al. including the directional correlation time delay for each individual, Highly correlated segments (HCS) for each pair and directed graph representation of the directional correaltion time dealy leadership heirarchy for each movement period  
  
* **Regularity.py**
  * Movement regularity feature extraction including PACS coordinate entropy for each individual at each movement period, Vector autoregression with and without exogenous varaibles for predicability of each individual's movement over time, and entropy measures for all features that are calculated over time
  




Tutorials

SampleData
SampleFeatures
