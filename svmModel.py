import os

import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt

_model = SVC(kernel='rbf', random_state=1)
is_trained = False
dest_output = './static/inference/graph'


def train_model(X, y):
    global is_trained
    train_sizes, train_scores, valid_scores = learning_curve(_model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

    plt.plot(train_sizes, np.mean(train_scores, axis=1), color='blue', label='train score')
    plt.plot(train_sizes, np.mean(valid_scores, axis=1), color='red', label='validation score')
    plt.xlabel("images")
    plt.ylabel("score")
    plt.legend()
    plt.savefig(os.path.join(dest_output, "loss_plot.png"))

    _model.fit(X, y)
    is_trained = True
    print("model trained")


def get_score(X, y):
    return _model.score(X, y)


def predict(testX):
    global is_trained
    if is_trained:
        return _model.predict(testX)
    return None
