import os
import random

import numpy as np
from flask import Flask, render_template_string, request
from sklearn.metrics import confusion_matrix

import svmModel
from utils import load_data, split, load_images, preprocess, predict

app = Flask(__name__)

dest = './static/inference/input'
dest_output = './static/inference/graph'


def get_hidden_file():
    file = open('./templates/index.html', 'r').read()
    return file.replace("####", "none")


def get_train_results(conf_mtx, train_score, test_score, loss_plot_path):
    info = f"Train score : %f <br> Test score : %f" % (train_score, test_score)
    data = open('./templates/index.html', 'r').read()
    data = data.replace("####", "block")
    data = data.replace("{train-visibility}", 'block')
    data = data.replace("{pred-visibility}", 'none')
    data = data.replace("{info}", info)
    data = data.replace("#Pred1#", str(conf_mtx[0][0]))
    data = data.replace("#Pred2#", str(conf_mtx[0][1]))
    data = data.replace("#Actual1#", str(conf_mtx[1][0]))
    data = data.replace("#Actual2#", str(conf_mtx[1][1]))
    data = data.replace("{loss_plot_path}", loss_plot_path)
    return data


def get_pred_results(isTumored: bool):
    file = open('./templates/index.html', 'r').read()
    file = file.replace("####", "block")
    file = file.replace("{train-visibility}", 'none')
    file = file.replace("{pred-visibility}", 'block')
    file = file.replace("====", "?dummy=" + str(random.randint(10000, 99990)))
    return file.replace("@@@@", "Brain Tumor is present" if isTumored else "Brain Tumor not present")


@app.route('/')
def index():
    return render_template_string(get_hidden_file())


@app.route('/process', methods=['POST'])
def process():
    file = request.files['file']
    if request.form.get('inference'):
        if file.filename != "":
            # prediction
            file.save(os.path.join(dest, file.filename))
            file.close()

            # predict using the model
            prediction = predict([os.path.join(dest, file.filename)])
            if prediction is None:
                return render_template_string(get_hidden_file())
            else:
                return render_template_string(get_pred_results(prediction[0] == 1))

    elif request.form.get('retrain'):
        # train the model
        dataset = load_data("dataset")
        print("loaded dataset...")

        x_train, y_train, x_test, y_test = split(dataset, 0.9)

        # ========================================
        # training
        # ========================================
        loaded_images = load_images(x_train)
        print("dataset split...")
        X = preprocess(loaded_images)
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
        y = y_train.to_numpy(dtype=np.ndarray).astype('int')
        print("prepared data...")

        # train model
        svmModel.train_model(X, y)
        train_score = svmModel.get_score(X, y)
        print("train score :", train_score)

        # ========================================
        # testing
        # ========================================
        loaded_images = load_images(x_test)
        X = preprocess(loaded_images)
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
        y = y_test.to_numpy(dtype=np.ndarray).astype('int')
        test_score = svmModel.get_score(X, y)
        print("test score: ", test_score)

        # confusion matrix
        y_pred = svmModel.predict(X)
        conf_mtx = confusion_matrix(y, y_pred)
        print("confusion matrix :\n", conf_mtx)

        # show train result
        return render_template_string(
            get_train_results(conf_mtx, train_score, test_score,
                              os.path.join(dest_output,
                                           "loss_plot.png?dummy=" + str(random.randint(10000, 99990))))
        )

    return render_template_string(get_hidden_file())


if __name__ == "__main__":
    app.run(port=5011, debug=True)
