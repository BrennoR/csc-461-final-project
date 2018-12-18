import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

data = pd.read_csv('train.csv')
data = data.drop(['Id', 'Soil_Type7', 'Soil_Type15'], axis=1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = pd.get_dummies(y).values

min_max_scaler = MinMaxScaler()
x2 = X[:, 10:]
x1 = min_max_scaler.fit_transform(X[:, :10])
X = np.concatenate((x1, x2), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = tf.keras.models.Sequential()

# Input Layer
model.add(tf.keras.layers.Dense(52, activation='relu', input_shape=(X.shape[1],)))

# Hidden Layer
model.add(tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None))

# Output Layer
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
results = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), verbose=False)
print("Accuracy:", np.mean(results.history["val_acc"]))
