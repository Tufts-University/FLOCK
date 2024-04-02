<h1 align="center" > FLOCK - Feature Learning of Organized Collective Kinetics </h1>

Welcome to the FLOCK-GPS repository

The purpose of this package is to serve as a processing and feature extraction toolkit for assessing group movement dynamics from GPS location data.


## Documentation
Documentation can be found in the [Docs](./Docs) folder of this repository


## Quick start

The [tutorials](./Tutorials) folder contains a walkthrough for each preprocessing and feature extraction step, while the [ExtractAllFeatures](./ExtractAllFeatures.py) script shows how to combine and run all processing and extraction. 
To run the predictive modeling analysis, fist instantiate your R home in the [Predictive-Modeling](./PredictiveModeling) notebook.


## FLOCK functions

* **DataLoading.py**
  * Functions for loading and re-formatting data from a directory of .csv files
  * See the SampleData folder for input data format examples
  * GPX data from different devices can be formatted uniquely, data is assumed to be in teh format of the example .csv files in the [SampleData](./SampleData) folder 
    
* **Preprocessing.py**
  * Preprocessing functions such as outlier sample detection, interpolation of missing datapoints, path smothing and break detection for identifying movement periods
    
* **VelocityFeats.py**
  * Feature extraction functions for finding the velocity of each individual over time and features related to difference in velocities across group members
  
* **SpatialFeats.py**
  * Feature extraction functions for spatio-temporal features (spatial features over time). Such as the stretch index, convex hull surface area and voronoi spaces
<p align="center">    
 <img src="Figures/SpatialFeatureFig.png" alt="Spatial feature figure" width="400" title="Spatial features" /> <br>
 <em>Spatial features such as distance to centroid (stretch index), convex hull surface area, and voronoi spaces</em>
</p>

* **PACS.py**
  * Path-adapted coordinate system transformation
<p align="center">       
 <img src="Figures/PACSfig.png" alt="PACS figure" width="600" title="PACS transformation" /> <br>
 <em>Path-adapted coordinate system example. The leftmost figure shows the raw path and one timepoint while the next figures show the straightened PACS path and coordinates of group members over a time-window in the PACS space.</em>
</p>

* **PacsFeats.py**
  * Feature extraction for path adapted coordinate system tranaformed data. Such as the spatial exploration index of each individual, the nearest neighbor (left/right and front/back), the length/width ratio of the group and the consistency of member positions in different movement periods
  
* **DirectionalCorrelation.py**
  * Directional correlation time delay leadership metrics from Nagy et. al. including the directional correlation time delay for each individual, Highly correlated segments (HCS) for each pair and directed graph representation of the directional correaltion time dealy leadership heirarchy for each movement period  
<p align="center">  
 <img src="Figures/DirCorrFig.png" alt="Directional correlation time delay figure" width="300" title="Directional correlation leadership heirarchy" /> <br>
 <em>Leadership heirarchy from the directional correlation time delay analysis</em>
</p>

* **ClusteringFeats.py**
  * Features from clustering analysis using a heirarchical density-based clustering method (HDBSCAN)


<p align="center">    
 <img src= 'Figures\GIF_Squad_1_0.gif' alt="Clustering gif" width="300" title="Clustering gif" /> <br>
 <em>HDBSCAN clustering in action, tick lines are 5m apart<br>Outliers are labelled cluster -1<br>Plotly express doesn't allow for categorical legends in scatter animations</em>
</p>

* **Regularity.py**
  * Movement regularity feature extraction including PACS coordinate entropy for each individual at each movement period, Vector autoregression with and without exogenous varaibles for predicability of each individual's movement over time, and entropy measures for all features that are calculated over time
  




Tutorials

SampleData
SampleFeatures
