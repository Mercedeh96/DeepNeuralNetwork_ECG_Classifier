import numpy as np
import pandas as pd
# import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from sklearn.metrics import accuracy_score
import keras
import os
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

number_of_classes = 19

df = pd.read_csv("2stage_train.csv", header=None)
x_train = df.values[:, :-1]
y_train = df.values[:, -1].astype(int)

# Validation dataset.
df = pd.read_csv("2stage_validate.csv", header=None)
x_validate = df.values[:, :-1]
y_validate = df.values[:, -1].astype(int)

# test dataset.
df = pd.read_csv("2stage_test.csv", header=None)
x_test = df.values[:, :-1]
y_test = df.values[:, -1].astype(int)

X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
X_val = np.reshape(x_validate, (x_validate.shape[0], 1, x_validate.shape[1]))
y_train = to_categorical(y_train)
y_validate = to_categorical(y_validate)
print("x train shape: ")
print(X_train.shape)
print("x val shape: ")
print(X_val.shape)
print("y_train shape")
print(y_train.shape)
print("y_validate shape")
print(y_validate.shape)
# y_train = y_train/x_train.max()
# create and fit the LSTM network
batch_size = 64
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(1, x_train.shape[1])))
# model.add(Dropout(0.25))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.25))
model.add(LSTM(64, return_sequences=True))
# model.add(Dropout(0.25))
model.add(LSTM(32))
# model.add(Dense(18, activation='relu'))
model.add(Dense(18, activation='relu'))
model.summary()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_val, y_validate), verbose=2,
          shuffle=False, callbacks=[early_stopping])
# model.save('Keras_models/my_model_' + str(i) + '_' + str(j) + '_' + str() + '.h5')
predictions = model.predict(X_val)
