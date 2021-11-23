import copy
import os

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error as mse

import svmModel

dataset_folder = './dataset/'
stage = './static/inference/stages/'
stage_1 = './static/inference/stages/1.jpg'
stage_2 = './static/inference/stages/2.jpg'
stage_3 = './static/inference/stages/3.jpg'
stage_4 = './static/inference/stages/4.jpg'
stage_5 = './static/inference/stages/5.jpg'


# loading up the data
def get_images_from_path(folder: str, cls: int):
    df = pd.DataFrame([], columns=["image", "class"])
    if os.path.isdir(folder) and os.path.isdir(folder):
        for file in os.listdir(folder):
            df = df.append({"image": os.path.join(folder, file), "class": cls}, ignore_index=True)
    return df


def load_data(data_path=""):
    if data_path is None or data_path == "":
        raise Exception("illegal file path")
    y = get_images_from_path(os.path.join(data_path, "yes"), 1)
    n = get_images_from_path(os.path.join(data_path, "no"), 0)
    df = pd.concat([y, n])
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def split(data_set, train_perc=0.8):
    if 0 < train_perc < 1:
        train_data = data_set.iloc[:int(data_set.shape[0] * train_perc - 1), :]
        test_data = data_set.iloc[int(data_set.shape[0] * train_perc):, :]
        return train_data["image"], train_data["class"], test_data["image"], test_data["class"]
    return None, None, None, None


def load_image(file: str):
    return np.array(Image.open(file, 'r').resize((224, 224)).convert('L')).astype('uint8')


def load_images(list_files):
    images = []
    for file in list_files:
        images.append(load_image(file))
    return np.array(images)


def denoise(image):
    blur = cv.GaussianBlur(image, (7, 7), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # Filter using contour area and remove small noise
    cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < 5500:
            cv.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Morph close and invert image
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return 255 - cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)


def contrast_stretch(image):
    return cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)


def skull_strip(image):
    # pass a single image
    ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    thresh = np.reshape(thresh, (thresh.shape[0], thresh.shape[1]))
    colormask = np.zeros((thresh.shape[0], thresh.shape[1]), dtype=np.uint8)
    colormask[thresh != 0] = 255
    blended = cv.addWeighted(image, 0.7, colormask, 0.1, 0)
    ret, markers = cv.connectedComponents(thresh)
    markers = np.reshape(markers, (markers.shape[0], markers.shape[1]))
    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
    if len(marker_area) > 0:
        largest_component = np.argmax(marker_area) + 1
    else:
        largest_component = [[]]
    brain_mask = markers == largest_component
    brain_out = image.copy()
    brain_out[brain_mask == False] = 0
    return brain_out


def preprocess(images_list):
    processed_image_list = []
    for image in images_list:
        res = contrast_stretch(image)
        res = skull_strip(res)
        res = threshold(res)
        res = denoise(res)
        processed_image_list.append(res)
    return np.array(processed_image_list)


def threshold(image, value=165):
    image[image <= value] = 0
    image[image > value] = 1
    return image


def predict(image_paths):
    images = load_images(image_paths)

    # original image
    Image.fromarray(images[0]).save(stage_1)

    # contrast stretch
    res = contrast_stretch(images[0])
    Image.fromarray(res).save(stage_2)

    # skull stripping
    res = skull_strip(res)
    Image.fromarray(res).save(stage_3)

    # thresholding
    res = threshold(res)
    Image.fromarray(res * 255).save(stage_4)

    res = denoise(res)
    Image.fromarray(res).save(stage_5)

    masked = copy.deepcopy(images[0])
    cv.cvtColor(masked, cv.COLOR_GRAY2RGB)
    a = 10

    input_data = np.reshape(res, (1, res.shape[0] * res.shape[1]))

    return svmModel.predict(input_data)
