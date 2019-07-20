# from sklearn import metrics
# from keras.layers import Dense, Embedding, SimpleRNN, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
# from keras.models import Sequential
# from keras_preprocessing import sequence
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from keras.optimizers import SGD
# opt = SGD(lr=0.01)
#
#
# df = pd.read_csv("train.csv", header=None)
# x_train = df.values[:, :-1]
# y_train = df.values[:, -1].astype(int)
#
#
# # Validation dataset.
# df = pd.read_csv("validate.csv", header=None)
# x_validate = df.values[:, :-1]
# y_validate = df.values[:, -1].astype(int)
#
# # test dataset.
# df = pd.read_csv("test.csv", header=None)
# x_test = df.values[:, :-1]
# y_test = df.values[:, -1].astype(int)
#
# max_features = 10
# maxlen = 500
# batch_size = 128
#
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x input_train shape: ', x_train.shape)
# print('x input_test shape', x_test.shape)
# print('y length: ', len(x_train))
# print('x length', len(y_train))
# print('y length: ', len(x_test))
# print('x length', len(y_test))
#
# # x_train = x_train.reshape(1, 87445, 500)
# # y_train = x_train.reshape(1, 87445, 500)
# # x_test = x_test.reshape(1, 29148, 500)
# # y_test = x_test.reshape(1, 29148, 500)
# model = Sequential()
# # model.add(Embedding(max_features, 64))
# # model.add(LSTM(512, return_sequences=True))
# # # model.add(Dropout(0.25))
# # model.add(LSTM(256, return_sequences=True))
# # # model.add(Dropout(0.25))
# # model.add(LSTM(128, return_sequences=True))
# # # model.add(Dropout(0.25))
# # model.add(LSTM(64, return_sequences=True))
# # # model.add(Dropout(0.25))
# # model.add(LSTM(32))
# # model.add(Dense(max_features, activation='softmax'))
#
# model.add(Embedding(max_features, 64))
# model.add(LSTM(64))
# # model.add(BatchNormalization())
# # model.add(Dropout(0.6))
# model.add(Dense(1, activation='relu'))
# # num_steps=500
#
# # hidden_size=500
# # model = Sequential()
# # model.add(Embedding(max_features, hidden_size, input_length=num_steps))
# # model.add(LSTM(hidden_size, return_sequences=True))
# # model.add(LSTM(hidden_size, return_sequences=True))
# # model.add(Dropout(0.5))
# # model.add(Dense(max_features, activation='softmax'))
# model.summary()
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# history = model.fit(x_test, y_test, epochs=3, batch_size=128, validation_split=0.2)
# mae = history.history['mae']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(mae) + 1)
# plt.plot(epochs, mae, 'bo', label='Training Acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label="Training Loss")
# plt.plot(epochs, val_loss, 'b', label="Validation loss")
# plt.title("Training and Validation loss")
# plt.legend()
# plt.show()
# # prediction_scores = model.predict(x_test, y_test, verbose=0)
# # validate_scores = model.evaluate(x_validate, y_validate, verbose=0)
# # print('prediction_scores')
# # print(prediction_scores)
# # print('validation scores')
# # print(validate_scores)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import accuracy_score
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

number_of_classes = 2


def change(x):
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)


df = pd.read_csv("train.csv", header=None)
x_train = df.values[:, :-1]
y_train = df.values[:, -1].astype(int)


# Validation dataset.
df = pd.read_csv("validate.csv", header=None)
x_validate = df.values[:, :-1]
y_validate = df.values[:, -1].astype(int)

# test dataset.
df = pd.read_csv("test.csv", header=None)
x_test = df.values[:, :-1]
y_test = df.values[:, -1].astype(int)

X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
X_val = np.reshape(x_validate, (x_validate.shape[0], 1, x_validate.shape[1]))

# create and fit the LSTM network
batch_size = 64
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(1, x_train.shape[1])))
#model.add(Dropout(0.25))
model.add(LSTM(256, return_sequences=True))
#model.add(Dropout(0.25))
model.add(LSTM(128, return_sequences=True))
#model.add(Dropout(0.25))
model.add(LSTM(64, return_sequences=True))
#model.add(Dropout(0.25))
model.add(LSTM(32))
model.add(Dense(1, activation='relu'))
model.summary()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_val, y_validate), verbose=2, shuffle=False, callbacks=[early_stopping])
# model.save('Keras_models/my_model_' + str(i) + '_' + str(j) + '_' + str() + '.h5')
predictions = model.predict(X_val)
