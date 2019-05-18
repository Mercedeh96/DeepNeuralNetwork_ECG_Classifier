import pandas as pd
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn import svm

# Training dataset.
df = pd.read_csv("train.csv", header=None)
x_train = df.values[:, :-1]
print(x_train)
y_train = df.values[:, -1].astype(int)
print(y_train)
# Validation dataset.
df = pd.read_csv("validate.csv", header=None)
x_validate = df.values[:, :-1]
y_validate = df.values[:, -1].astype(int)
print(y_validate)

# Test dataset.
df = pd.read_csv("test.csv", header=None)
x_test = df.values[:, :-1]
y_test = df.values[:, -1].astype(int)
print(y_test)

# # visualise data
# C0 = np.argwhere(y_train == 0).flatten()
# C1 = np.argwhere(y_train == 1).flatten()
#
# x = np.arange(0, 187)*8/1000.0
#
# plt.figure(figsize=(20,12))
# plt.plot(x, x_train[C0, :][0], label="Normal") # Display first normal beat.
# plt.plot(x, x_train[C1, :][0], label="Abnormal") # Display first abnormal beat.
# plt.legend()
# plt.title("1-beat ECG for every category", fontsize=20)
# plt.ylabel("Normalized Amplitude (0 - 1)", fontsize=15)
# plt.xlabel("Time (ms)", fontsize=15)
# plt.show()

# # training the model using svm
# clf = svm.SVC(gamma='scale', kernel='rbf')
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# # print(y_pred [0:5])
# # Model Accuracy: how often is the classifier correct?
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Create a svm Classifier
clf = svm.SVC(gamma='scale', kernel='rbf')
print("this line shows the clf svm ")
print(clf)

# Train the model using the training sets
clf.fit(x_train, y_train)
print("this is the line for clf.fit")
print(clf.fit(x_train, y_train))

# Predict the response for test dataset
y_pred = clf.predict(x_test)
print(y_pred)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))