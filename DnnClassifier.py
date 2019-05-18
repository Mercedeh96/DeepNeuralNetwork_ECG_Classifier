import pandas as pd
import numpy as np
# from sklearn import metrics
# from keras.layers import Dense, Embedding, SimpleRNN, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
# from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_estimator as tf1
# from google.datalab.ml import TensorBoard

# from keras_preprocessing import sequence
#
# max_features = 10000
# maxlen = 500
# batch_size = 32
# Training dataset.
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


#
# print("loading data...")
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('input_train shape: ', x_train.shape)
# print('input_test shape', x_test.shape)
#
# model = Sequential()
# model.add(Embedding(max_features, 64))
# model.add(LSTM(64))
# model.add(Dropout(0.6))
# model.add(Dense(1, activation='relu'))
# # model.add(SimpleRNN(32))
# model.summary()
#
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(x_test, y_test, epochs=3, batch_size=128, validation_split=0.2)
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training Acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label="Training Loss")
# plt.plot(epochs, val_loss, 'b', label="Validation loss")
# plt.title("Training and Validation loss")
# plt.legend()
# plt.show()

# data visualisation
C0 = np.argwhere(y_train == 0).flatten()
C1 = np.argwhere(y_train == 1).flatten()
x = np.arange(0, 187)*8/1000.0

plt.figure(figsize=(20,12))
plt.plot(x, x_train[C0, :][0], label="Normal") # Display first normal beat.
plt.plot(x, x_train[C1, :][0], label="Abnormal") # Display first abnormal beat.
plt.legend()
plt.title("1-beat ECG for every category", fontsize=20)
plt.ylabel("Normalized Amplitude (0 - 1)", fontsize=15)
plt.xlabel("Time (ms)", fontsize=15)
# plt.show()
# training the model

feature_columns = [tf.feature_column.numeric_column('beat', shape=[187])]
print(feature_columns)
estimator = tf1.estimator.DNNClassifier(
   feature_columns=feature_columns,
   hidden_units=[256, 64, 16],
   optimizer=tf.train.AdamOptimizer(1e-4),
   n_classes=2,
   dropout=0.1,
   model_dir='ecg_model'
)
print("estimator is working")

input_fn_train = tf1.estimator.inputs.numpy_input_fn(
    x={'beat': x_train},
    y=y_train,
    num_epochs=None,
    batch_size=50,
    shuffle=True
)
print("input fn train is working")
estimator.train(input_fn=input_fn_train, steps=400000)
# model validation
input_fn_validate = tf1.estimator.inputs.numpy_input_fn(
    x={'beat': x_validate},
    y=y_validate,
    num_epochs=1,
    shuffle=False
)
accuracy_score = estimator.evaluate(input_fn=input_fn_validate)
print('\nTest Accuracy: {0:f}%\n'.format(accuracy_score['accuracy']*100))

# testing the model
input_fn_test = tf1.estimator.inputs.numpy_input_fn(
 x={'beat': x_test},
 y=y_test,
 num_epochs=1,
 shuffle=False
)
predictions = estimator.predict(input_fn=input_fn_test)
totvals = 0
totwrong = 0

for prediction, expected in zip(predictions, y_test):
    totvals = totvals + 1
    catpred = prediction['class_ids'][0]
    certainty = prediction['probabilities'][catpred] * 100
    if (expected != catpred):
        totwrong = totwrong + 1
        #print (prediction)
        print('Real: ', expected, ', pred: ', catpred, ', cert: ', certainty)

print('Accuracy: ', ((totvals - totwrong) * 100.0 / totvals))
print('Wrong: ', totwrong, ' out of ', totvals)

# Monitoring with TensorBoard
# TensorBoard().start('ecg_model')
# TensorBoard().list()

# Stop TensorBoard
# for pid in TensorBoard.list()['pid']:
#     TensorBoard().stop(pid)
#     print('Stopped TensorBoard with pid ', pid)

# exporting the model
# Build receiver function, and export.
feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
serving_input_receiver_fn = tf1.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

feature_placeholders = {'beat': tf.placeholder(dtype=tf.float32, shape=(187,)) }
serving_input_receiver_fn = tf1.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders)
export_dir = estimator.export_savedmodel('ecg_serving', serving_input_receiver_fn, strip_default_attrs=True)










