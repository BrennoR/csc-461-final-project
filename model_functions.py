import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import csv


def init_data(normalize=True):
    """
    Initializes the data.
    """
    data = pd.read_csv('train.csv')
    data = data.drop(['Id', 'Soil_Type7', 'Soil_Type15'], axis=1)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    if normalize:
        min_max_scaler = MinMaxScaler()
        x2 = X[:, 10:]
        x1 = min_max_scaler.fit_transform(X[:, :10])
        X = np.concatenate((x1, x2), axis=1)

    return X, y


actual_list = []
predict_list = []


def model_analysis(model, X, y):
    """
    Perform cross validation and score analysis for different algorithms.
    """
    model = model

    print("=" * 100, "\n", str(model), "\n", "=" * 100)

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    accuracies = []

    # Cross-Validation
    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = accuracy_score(y_test, y_pred)
        accuracies.append(score)
        actual_list.append(list(y_test))
        predict_list.append(list(y_pred))

    accuracy = np.average(accuracies)
    print("Accuracy Score:", accuracy)


X, y = init_data()

# Hyperparameter Testing

for C in [0.01, 0.1, 1, 10, 100]:
    svm = SVC(C=C, kernel='rbf')
    model_analysis(svm, X, y)

actual_list = sum(actual_list, [])
predict_list = sum(predict_list, [])
cm = confusion_matrix(actual_list, predict_list)

# test_list = [actual_list, predict_list]

values = []
for i in range(len(actual_list)):
    values.append([actual_list[i], predict_list[i]])

with open('submission', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(values)

# Confusion Matrix Heatmap

plt.figure(figsize=(8,6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, cmap='Greens', fmt='g')
plt.xlabel('Predicted Class', fontsize=14)
plt.ylabel('Actual Class', fontsize=14)
plt.title('Logistic Regression - Confusion Matrix', fontsize=15)
plt.show()
