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
from the data. The first column on the data was also dropped as it simply contained the instance ID and provided no useful information for prediction.

A correlation heatmap was plotted for the non-categorical features in order to check if any features were highly correlated. The heatmap is shown below:

<a href="url"><img src="https://user-images.githubusercontent.com/31149320/50188150-8fd3b880-02ee-11e9-9d46-b00390d2c62f.png" align="center" height="560" width="700" ></a>

As shown above, Hillshade_9am and Hillshade_3pm, Hillshade_noon and Slope, and Aspect and Hillshade_9am, Horizontal_Distance_To_Hydrology and Vertical_Distance_To_Hydrology, Aspect and Hillshade_3pm, Hillshade_noon and Hillshade_3pm, and Elevation and Horizontal_Distance_To_Roadways all displayed fairly strong correlations. This information could have been used to perform feature selection come modeling time.

The relationships between Wilderness Area, Soil Type and Cover Type were then plotted on histograms to examine the class distribution for these features. The plots are shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/31149320/50188664-556b1b00-02f0-11e9-960f-6ff46d0cc5a2.png">
</p>

<p align="center">
  <img width="448" height="360" src="https://user-images.githubusercontent.com/31149320/50188678-5e5bec80-02f0-11e9-8788-de242d1c3f0d.png">
</p>

As shown above, some classes have great distinction with different Soil Types and Wilderness Areas while others not so much.

Preprocessing was then performed on the non-categorical featuress up to Wilderness Area. The MinMaxScaler from sklearn's preprocessing module was used for the task. This scaling shifted the range of the features to between zero and one.

## Models & Algorithms

## k-Nearest Neighbors

k-Nearest Neighbors was the first algorithm implemented for prediction. The KNeighborsClassifier class from sklearn was used for this. Hyperparameter selection was made using a 5-fold cross validation study. Three parameters were considered for the study: number of neighbors, distance metric, and weighting metric. The final parameters were chosen to be 3 neighbors, euclidean distance, and a distance weight metric based on the cross validation results. An accuracy of 76.83 % was achieved. The resulting confusion matrix from a 5-fold cross validation run is shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/31149320/50188664-556b1b00-02f0-11e9-960f-6ff46d0cc5a2.png">
</p>

## Logistic Regression

Logistic Regression was then implemented by also utilizing the respective sklearn class. Only the penalty and C value hyperparameters were tuned for this study. Hyperparameter selection was performed in the same way as in k-Nearest Neighbors and the best parameters were shown to be a C value of 10 with a 'l1' penalty scheme. An accuracy of 68.44 % was achieved. The resulting confusion matrix from a 5-fold cross validation run is shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/31149320/50188664-556b1b00-02f0-11e9-960f-6ff46d0cc5a2.png">
</p>

## Support Vector Machine

Support Vector Machines are a very popular and succesful machine learning algorithm and as such we were very optimistic in utilizing them on the dataset. Similar to both Logistic Regression and k-Nearest Neighbors, 5-fold cross validation was used to perform hyperparameter selection. The hyperparameters considered where the penalty value of C, the kernel (gaussian or poly), and the degree (only applicable to poly kernel). The final parameters selected where a hard-margin penalty value of 1e5 along with the gaussian kernel as this combination performed the best in validation. An accuracy of 76.11 % was achieved. The resulting confusion matrix from a 5-fold cross validation run is shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/31149320/50188664-556b1b00-02f0-11e9-960f-6ff46d0cc5a2.png">
</p>

## Neural Network

Based on the success of neural networks and their increasing popularity in recent times, neural networks were also chosen as a potential model to test on the dataset. The neural network was implemented using keras and computations were made with the help of a GPU running nVIDIA's CUDA technology. A two layer neural network with one hidden layer model utilizing cross-entropy loss and the adam optimizer was considered for this study. A model of the network is shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/31149320/50188664-556b1b00-02f0-11e9-960f-6ff46d0cc5a2.png">
</p>

The only parameters considered for the neural network in this study were the number of nodes in the hidden layer and the batch size used with the adam optimizer. Many tests were run varying these parameters and a batch size of 32 with a hidden layer size of 50 were chosen for the final network configuration. An accuracy of 81.35 % was achieved. The resulting confusion matrix obtained from the validation set is shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/31149320/50188664-556b1b00-02f0-11e9-960f-6ff46d0cc5a2.png">
</p>

## Test Submission and Conclusion

The various algorithms with their respective best hyperparameters were then used on the Kaggle test data and submitted to the site. The results were shocking. The neural network which performed the best in training and validation, only achieved an accuracy of 65.57 %! The simplest algorithm, k-Nearest Neighbors, outperformed the neural network by one-hundredth of a percent with a final accuracy of 65.68 %. Logistic Regression was extremely dissapointing with a score of 52.70 %. The Support Vector Machine performed the best out of all achieving an accuracy of 67.67 %. The results were not what we expected at all. It is quite possible that significant overfitting occured with the Neural Network. The algorithms chosen could have also been simply not suited for the task as well some others. Later tests were run using a Random Forest and better results were achieved without any hyperparameter tuning. Another study could be performed using other methods such as Random Forests to see if satisfactory results could be achieved with these other methods!
