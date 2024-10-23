# --------------------
#   Imports
# --------------------

import os                                           # standard
import sys
import random
from math import sqrt                               # math and analysis
import numpy as np
import pandas as pd
from scipy.signal import convolve2d, medfilt        # signal processing
from scipy.ndimage import gaussian_filter
import pywt
import cv2                                          # image processing
from PIL import Image
from skimage.transform import rescale
import matplotlib                                   # visualization
import matplotlib.pyplot as plt

# --------------------
#   Edge detection
# --------------------

def sobel_edges(img):
    sobelimage = img.copy()
    sobelx = cv2.Sobel(sobelimage, cv2.CV_64F, 1, 0, ksize = 3)
    sobely = cv2.Sobel(sobelimage, cv2.CV_64F, 0, 1, ksize = 3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = cv2.convertScaleAbs(magnitude)
    _, thresholded = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
    return thresholded
  
def canny_edges(img):
    th1 = 30
    th2 = 60
    d = 2
    edgeresult = img.copy()
    edgeresult = cv2.GaussianBlur(edgeresult, (2*d + 1, 2*d + 1), -1)[d:-d, d:-d]
    edgeresult = edgeresult.astype(np.uint8)
    edges = cv2.Canny(edgeresult, th1, th2)
    return edges

# --------------------
#   Global attacks
# --------------------

def blur_gauss(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

def blur_median(img, size):
    attacked = medfilt(img, size)
    return attacked

def awgn(img, mean, std, seed):
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def jpeg_compression(img, qf):
    img = Image.fromarray(img)
    img = img.convert('L') 
    img.save('tmp.jpg', "JPEG", quality = qf)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype = np.uint8)
    os.remove('tmp.jpg')
    return attacked

def resize(img, scale):
    x, y = img.shape
    attacked = rescale(img, scale, anti_aliasing = True, mode = 'reflect')
    attacked = rescale(attacked, 1 / scale, anti_aliasing = True, mode = 'reflect')
    attacked = np.asarray(attacked * 255, dtype = np.uint8)
    attacked = cv2.resize(attacked, (y, x), interpolation = cv2.INTER_LINEAR)
    return attacked

# --------------------
#   Localized attacks
# --------------------

def blur_edge(img, sigma, edge_func, blur_func):
    edges = edge_func(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    blurred_img = blur_func(img, sigma)
    attacked = (1 - edges) * img + edges * blurred_img
    return attacked

def awgn_edge(img, std, seed, edge_func):
    attacked = img.copy()
    edge_res = img.copy()
    global_awgn = awgn(img, std, seed)
    edges = edge_func(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    edge_res[edges > 0] = [255]
    attacked[edges > 0] = global_awgn[edges > 0] 
    return attacked

# --------------------
#   Tranform attacks
# --------------------

# --------------------
#   Combo attacks
# --------------------

def resize_jpeg(img, qf, scale):
    compressed = jpeg_compression(img, qf)
    attacked = resize(compressed, scale)
    return attacked