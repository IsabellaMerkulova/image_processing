from scipy.linalg import norm
from utils import timeit
import cv2
from numba import jit, vectorize


@jit(nopython=False)
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng


@jit(nopython=False)
def get_diff(img1, img2):
    return img1 - img2  # elementwise for scipy arrays


@timeit
def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = get_diff(img1, img2)
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)


img1 = cv2.cvtColor(cv2.imread("images/801_a_g_1.png"), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread("images/801_a_s_1.png"), cv2.COLOR_BGR2GRAY)
r = compare_images(img1, img2)

