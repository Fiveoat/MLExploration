import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
import numpy as np

data = pd.read_csv('diabetes.csv')
labels = data['Outcome']
features = data.iloc[:, 0:8]

x = features

y = np.ravel(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)

scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(8, activation='relu', input_shape=(8,)))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=5, verbose=1, validation_split=.1)

y_pred = np.argmax(model.predict(x_test), axis=-1)
print(y_pred)

score = model.evaluate(x_test, y_test, verbose=1)

print(score)
