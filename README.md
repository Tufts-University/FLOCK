APPROVED FOR PUBLIC RELEASE: PR2024-1237


<h1 align="center" > FLOCK - Feature Learning of Organized Collective Kinetics </h1>

Welcome to the FLOCK-GPS repository

The purpose of this package is to serve as a processing and feature extraction toolkit for assessing group movement dynamics from GPS location data.


## Quick start

### Installation

`pip install FLOCK-GPS` 

FLOCK-GPS for Python >= 3.7 and all its dependancies are available from PyPi

### Tutorials

To run tutorials, download the repository to your local storage and open the notebooks from there.

The [Tutorials](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials) folder contains a walkthrough for each preprocessing and feature extraction step, while the [ExtractAllFeatures](https://github.com/Tufts-University/FLOCK/blob/main/ExtractAllFeatures.py) script shows how to combine and run all processing and extraction. 
To run the predictive modeling analysis, fist instantiate your R home in the [Predictive-Modeling](https://github.com/Tufts-University/FLOCK/blob/main/PredictiveModeling) notebook.


### Documentation
Documentation can be found in the [Docs](https://github.com/Tufts-University/FLOCK/blob/main/Docs) folder of this repository. The HTML files will open in your default browser from the donloaded repository.


## FLOCK functions

* **DataLoading.py** 
  * Functions for loading and re-formatting data from a directory of .csv files
  * See the SampleData folder for input data format examples
  * GPX data from different devices can be formatted uniquely, data is assumed to be in teh format of the example .csv files in the [SampleData](https://github.com/Tufts-University/FLOCK/blob/main/SampleData) folder 
    
* **Preprocessing.py** [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials/Break-Detection.ipynb)
  * Preprocessing functions such as outlier sample detection, interpolation of missing datapoints, path smothing and break detection for identifying movement periods
    
* **VelocityFeats.py** [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials/Velocity-Metrics.ipynb)
  * Feature extraction functions for finding the velocity of each individual over time and features related to difference in velocities across group members
  
* **SpatialFeats.py** [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials/Spatial-Metrics.ipynb)
  * Feature extraction functions for spatio-temporal features (spatial features over time). Such as the stretch index, convex hull surface area and voronoi spaces
<p align="center">    
 <img src="https://github.com/Tufts-University/FLOCK/blob/main/Figures/SpatialFeatureFig.png" alt="Spatial feature figure" width="400" title="Spatial features" /> <br>
 <em>Spatial features such as distance to centroid (stretch index), convex hull surface area, and voronoi spaces</em>
</p>

* **PACS.py** [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials/PACS-Metrics.ipynb)
  * Path-adapted coordinate system transformation
<p align="center">       
 <img src="https://github.com/Tufts-University/FLOCK/blob/main/Figures/PACSfig.png" alt="PACS figure" width="600" title="PACS transformation" /> <br>
 <em>Path-adapted coordinate system example. The leftmost figure shows the raw path and one timepoint while the next figures show the straightened PACS path and coordinates of group members over a time-window in the PACS space.</em>
</p>

* **PacsFeats.py** [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials/PACS-Metrics.ipynb)
  * Feature extraction for path adapted coordinate system tranaformed data. Such as the spatial exploration index of each individual, the nearest neighbor (left/right and front/back), the length/width ratio of the group and the consistency of member positions in different movement periods
  
* **DirectionalCorrelation.py** [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials/Leadership-Metrics.ipynb)
  * Directional correlation time delay leadership metrics from [Nagy et. al.](https://www.nature.com/articles/nature08891) including the directional correlation time delay for each individual, Highly correlated segments (HCS) for each pair and directed graph representation of the directional correaltion time dealy leadership heirarchy for each movement period  
<p align="center">  
 <img src="https://github.com/Tufts-University/FLOCK/blob/main/Figures/DirCorrFig.png" alt="Directional correlation time delay figure" width="300" title="Directional correlation leadership heirarchy" /> <br>
 <em>Leadership heirarchy from the directional correlation time delay analysis</em>
</p>

* **ClusteringFeats.py** [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials/Clustering-Metrics.ipynb)
  * Features from clustering analysis using a density-based clustering method (DBSCAN) and/or a heirarchical density-based clustering method (HDBSCAN)


<p align="center">    
 <img src= 'https://github.com/Tufts-University/FLOCK/blob/main/Figures/GIF_Squad_1_0.gif' alt="Clustering gif" width="300" title="Clustering gif" /> <br>
 <em>HDBSCAN clustering in action, tick lines are 5m apart<br>Outliers are labelled cluster -1<br>Plotly express doesn't allow for categorical legends in scatter animations</em>
</p>

* **Regularity.py** [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/Tutorials/Regularity-Metrics.ipynb)
  * Movement regularity feature extraction including PACS coordinate entropy for each individual at each movement period, Vector autoregression with and without exogenous varaibles for predicability of each individual's movement over time, and entropy measures for all features that are calculated over time
 

## Predictive modelling [![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/Tufts-University/FLOCK/blob/main/PredictiveModeling/Predictive-Modeling.ipynb)
Case study: During a 72 hour field study, 16 groups of soldiers complete a 'loaded ruck march' before engaging in strategic and tactical exercises the next day. During these exercises, experts score each of the groups. Our goal is to predict the expert's score of each group from features calculated with this toolbox during the loaded ruck march. To select features, we employ the [Terminating Random Experiments (TRex)](https://github.com/jasinmachkour/TRexSelector) method. We then train a linear regression model using only the selected features.

* **Explanatory model**
  * Here we use all of our datapoints (16 groups) to select important features and we report the model's performance on the training data. This exploratory model achieves an R squared score of 0.82.
<p align="center">       
 <img src="https://github.com/Tufts-University/FLOCK/blob/main/Figures/ExplanatoryModelResults.png" alt="Explanatory Model Results" width="300" title="Explanatory Model Results" /> <br>
 <em>Explanatory Model Results</em>
</p>

* **Predictive model**
  * Here we employ a Leave-One-Out cross validation method, where we select features and train a new model for each iteration, then test the model on the left out datapoint. This predictive model achieves an R squared score of 0.63.
<p align="center">       
 <img src="https://github.com/Tufts-University/FLOCK/blob/main/Figures/PredictiveModelResults.png" alt="Predictive Model Results" width="300" title="Predictive Model Results" /> <br>
 <em>Predictive Model Results</em>
</p>

**Why the T-Rex selector?** <br>
Compared with a basic Least Absolute Shrinkage and Selection Operator (LASSO) feature selection method, the TRex achieves a much higher stability (i.e. the selcted features in each Leave-One-Out iteration are more similar). The LASSO struggles with wide datasets (small number of observations and large number of features), especially when there is a low sparsity (a high amount of relevant features) such as with our sample study dataset.


## Citation information

Please cite [[0]](#publications) when using FLOCK-GPS in your research and reference the appropriate release version.

## Publications

### About FLOCK-GPS

[0]

APPROVED FOR PUBLIC RELEASE: PR2024-1237

