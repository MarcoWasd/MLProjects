import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from tensorflow import keras

# Data loading and preparation
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]

X = X.to_numpy()
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Side pixels are always white, so they do not carry information to the classifier

X_train_cut = np.empty([60000,400])
for i in range(60000):
    X_train_cut[i] = X_train[i].reshape(28,28)[5:25,5:25].reshape(400,) 

X_test_cut = np.empty([10000,400])
for i in range(10000):
    X_test_cut[i] = X_test[i].reshape(28,28)[5:25,5:25].reshape(400,) 

# Feature scaling
scaler = StandardScaler()
X_train_scaled_cut = scaler.fit_transform(X_train_cut.astype(np.float64))

# First element is a number five. I will start with a classifier for the number five
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Stocastic Gradiend Classifier for digit 5
sgd = SGDClassifier(random_state=42, n_jobs=-1)
sgd.fit(X_train_cut, y_train_5)

y_train_5_pred = cross_val_predict(sgd, X_train_scaled_cut, y_train_5, cv=3)

confusion_matrix(y_train_5, y_train_5_pred)

# Multi class Classifier for all classes of numbers from 0 to 9

svc_cut = SVC()
SVC_score = cross_val_score(svc_cut, X_train_scaled_cut, y_train, cv=3, scoring="accuracy", n_jobs=-1)

# Multiclass Classifier with KNeighborsClassifier

kn = KNeighborsClassifier(n_jobs=-1)
#kn.fit(X_train_scaled, y_train)
#kn_score = cross_val_score(kn, X_train, y_train,  cv=3, scoring="accuracy")

# Grid search to optimize hyperparameters
k_range = list(range(2, 20))
param_grid = {"n_neighbors":k_range, "weights":['distance', 'uniform']}

grid = GridSearchCV(kn, param_grid, cv=3, scoring='accuracy', n_jobs=18)
grid.fit(X_train_scaled_cut, y_train)

final_model = grid.best_estimator_
predictions = final_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions)

# Neural Network

model = keras.models.Sequential()

model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(X_train_scaled_cut, y_train, epochs=30)

model.evaluate(X_test_cut, y_test)
















