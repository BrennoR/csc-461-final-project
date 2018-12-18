# CSC 461 (Machine Learning) - Final Project

This was the final project for the CSC461 machine learning course at URI. We were tasked to apply machine learning algorithms 
to a dataset of our choice on Kaggle. My group decided on the [Forest Cover Type Prediction Competition](https://www.kaggle.com/c/forest-cover-type-prediction) 
as it seemed a decent level task for our experience at the time.
We decided on utilizing four different machine learning algorithms and seeing how well they perform on this dataset.
The four algorithms/models chosen were: k-Nearest Neighbors, Logistic Regression, Support Vector Machines, and a Neural
Network.

## Data Exploration and Preprocessing

The first step was to explore the data and perform any preprocessing if necessary. The data attributes/features were first 
checked and are shown below:
  - **Elevation -** Elevation in meters
  - **Aspect -** Aspect in degrees azimuth
  - **Slope -** Slope in degrees
  - **Horizontal_Distance_To_Hydrology -** Horz Dist to nearest surface water features
  - **Vertical_Distance_To_Hydrology -** Vert Dist to nearest surface water features
  - **Horizontal_Distance_To_Roadways -** Horz Dist to nearest roadway
  - **Hillshade_9am (0 to 255 index) -** Hillshade index at 9am, summer solstice
  - **Hillshade_Noon (0 to 255 index) -** Hillshade index at noon, summer solstice
  - **Hillshade_3pm (0 to 255 index) -** Hillshade index at 3pm, summer solstice
  - **Horizontal_Distance_To_Fire_Points -** Horz Dist to nearest wildfire ignition points
  - **Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) -** Wilderness area designation
  - **Soil_Type (40 binary columns, 0 = absence or 1 = presence) -** Soil Type designation
  - **Cover_Type (7 types, integers 1 to 7) -** Forest Cover Type designation
  
As shown above, the features are all categorical starting at Wilderness Area. This signifies that any scaling or normalization
should only be applied on the features up to Wilderness Area. The class distribution on the data was then checked and it 
was shown that each class has exactly 2160 instances in the training data. Null values were checked and fortunately none were
found. Constant features were then checked for. Soil Type 7 and 15 were shown to be constant features and as such were dropped
from the data.

A correlation heatmap was plotted for the non-categorical features in order to check if any features were highly correlated. The heatmap is shown below:

<a href="url"><img src="https://user-images.githubusercontent.com/31149320/50188150-8fd3b880-02ee-11e9-9d46-b00390d2c62f.png" align="center" height="560" width="700" ></a>

As shown above, Hillshade_9am and Hillshade_3pm, Hillshade_noon and Slope, and Aspect and Hillshade_9am, Horizontal_Distance_To_Hydrology and Vertical_Distance_To_Hydrology, Aspect and Hillshade_3pm, Hillshade_noon and Hillshade_3pm, and Elevation and Horizontal_Distance_To_Roadways all displayed fairly strong correlations. This information could have been used to perform feature selection come modeling time.

The relationships between Wilderness Area, Soil Type and Cover Type were then plotted on histograms to examine the class distribution for these features. The plots are shown below:


