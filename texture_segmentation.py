import logging

import cv2
from math import e
import matplotlib.pyplot as plt
from numba import jit, vectorize
import numpy as np
import scipy.ndimage as ndi
from utils import timeit
from skimage.feature.texture import greycoprops, greycomatrix


logger = logging.getLogger()
logger.setLevel('INFO')


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


def get_uniques(window):
    return np.unique(window, return_counts=True)

@jit(nopython=False)
@vectorize(['float32(float32, float32)'], target='cuda')
def count_entropy(counts):
    norm_counts = counts / sum(counts)
    return -(norm_counts * np.log(norm_counts) / np.log(e)).sum()

def get_entropy(window, base=None):
    counts = get_uniques(window)
    return count_entropy(counts)

#@vectorize(['float32(float32, float32)'], target='cuda')
def get_smoothness(window):
    # gives an array
    return np.average(np.absolute(ndi.filters.laplace(window)))

@jit(nopython=False)
def get_smoothness2(window):
    # IT does not work
    return 1 - 1/(1+np.var(window))

@timeit
def get_result_matrix(image, step_size, window_size, param_func):
    logging.info(param_func)
    param_matrix = np.zeros(image.shape, np.uint8)

    for x, y, window in sliding_window(
            image, stepSize=step_size, windowSize=window_size
    ):
        param_matrix[y][x] = param_func(window)
    return param_matrix


@jit(nopython=False)
@vectorize(['float32(float32, float32)'], target='cuda')
def get_var(window):
    return np.var(window)


@jit(nopython=False)
@vectorize(['float32(float32, float32)'], target='cuda')
def get_std(window):
    return np.std(window)


@jit(nopython=False)
#@vectorize(['float32(float32, float32)'], target='cuda')
def get_mean(window):
    return np.mean(window)


def get_homogeneity(window):
    angles = [0, np.pi / 2]
    glcm = greycomatrix(window, [1], [np.pi/2])
    homogeneity = greycoprops(glcm, 'homogeneity')
    return homogeneity[0][0]
    # props = ['contrast', 'dissimilarity', 'homogeneity']


def get_sliding_window_properties(input_image, sliding_window_method=True):
    parameters = {
       'variance': get_var,  # дисперсия
       'std': get_std,  # СКО
       'mean': get_mean,  # среднее
       'entropy': get_entropy,  # энтропия
       'smoothing': get_smoothness,  # энтропия
    }
    fig = plt.figure(figsize=(8, 8))
    logging.info(f'Processing file {input_image}')
    window_sizes = [(4, 4), (8, 8), (16, 16)]
    step_size = 1
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    rows, columns = len(window_sizes), len(parameters)
    i = 1
    params_time = {}
    params_avg = {}
    for window_size in window_sizes:
        params_time[str(window_size)] = {}
        params_avg[str(window_size)] = {}
        if not sliding_window_method:
            step_size = window_size[0]
        for param_key, param_func in parameters.items():
            param_matrix, time_consumed = get_result_matrix(image, step_size, window_size, param_func)
            params_time[str(window_size)][param_key] = round(time_consumed, 2)
            ax = fig.add_subplot(rows, columns, i)
            plt.imshow(param_matrix)
            params_avg[str(window_size)][param_key] = round(param_matrix.mean(), 2)
            ax.title.set_text(f'{param_key}_{window_size}')
            i += 1
    # # third moment
    # third_moment_matrix = get_third_moment(image)
    # ax = fig.add_subplot(rows, columns, i)
    # plt.imshow(third_moment_matrix)
    # ax.title.set_text(f'third moment matrix')
    # print(params_avg)
    print(params_time)
    fig.suptitle(input_image)
    plt.show()

    return image


def show_result_images(result):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 1
    for i in range(1, len(result) + 1):
        img = result[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()




# For method using "setka" use False in second argument
# get_sliding_window_properties('images/test.png', False)


# TODO: check
#  from skimage.feature.texture import greycomatrix, greycoprops
# glcm = greycomatrix(window, d, theta, levels)
# contrast = greycoprops(glcm, 'contrast')
# props = ['contrast', 'dissimilarity', 'homogeneity']