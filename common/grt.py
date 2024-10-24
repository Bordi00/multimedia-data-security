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
from scipy.fft import dct, idct
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
    args = {key: value for key, value in list(locals().items())[1:]}
    attacked = gaussian_filter(img, sigma)
    return (attacked, blur_gauss.__name__, args)

def blur_median(img, size):
    args = {key: value for key, value in list(locals().items())[1:]}
    attacked = medfilt(img, size)
    return (attacked, blur_median.__name__, args)

def awgn(img, mean, std, seed):
    args = {key: value for key, value in list(locals().items())[1:]}
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return (attacked, awgn.__name__, args)

def jpeg_compression(img, qf):
    args = {key: value for key, value in list(locals().items())[1:]}
    img = Image.fromarray(img)
    img = img.convert('L') 
    img.save('tmp.jpg', "JPEG", quality = qf)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype = np.uint8)
    os.remove('tmp.jpg')
    return (attacked, jpeg_compression.__name__, args)

def resize(img, scale):
    args = {key: value for key, value in list(locals().items())[1:]}
    if scale == 1:
        return (img, resize.__name__, args)
    x, y = img.shape
    attacked = rescale(img, scale, anti_aliasing = True, mode = 'reflect')
    attacked = rescale(attacked, 1 / scale, anti_aliasing = True, mode = 'reflect')
    attacked = np.asarray(attacked * 255, dtype = np.uint8)
    attacked = cv2.resize(attacked, (y, x), interpolation = cv2.INTER_LINEAR)
    return (attacked, resize.__name__, args)

# --------------------
#   Localized attacks
# --------------------

def blur_edge(img, blur_func, sigma, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    edges = edge_func(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    blurred_img = blur_func(img, sigma)[0]
    attacked = (1 - edges) * img + edges * blurred_img
    return (attacked, blur_edge.__name__, args)

def blur_flat(img, blur_func, sigma, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    edges = edge_func(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    blurred_img = blur_func(img, sigma)[0]
    attacked = edges * img + (1 - edges) * blurred_img
    return (attacked, blur_flat.__name__, args)

def awgn_edge(img, mean, std, seed, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    attacked = img.copy()
    edge_res = img.copy()
    global_awgn = awgn(img, mean, std, seed)[0]
    edges = edge_func(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    edge_res[edges > 0] = [255]
    attacked[edges > 0] = global_awgn[edges > 0]
    return (attacked, awgn_edge.__name__, args)

# --------------------
#   Tranform attacks
# --------------------

def blur_dwt(img, blur_func, sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_blurred = blur_func(LL, sigma[0])[0]
    LH_blurred = blur_func(LH, sigma[1])[0]
    HL_blurred = blur_func(HL, sigma[2])[0]
    HH_blurred = blur_func(HH, sigma[3])[0]    
    coeffs_blurred = (LL_blurred, (LH_blurred, HL_blurred, HH_blurred))
    attacked = pywt.idwt2(coeffs_blurred, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, blur_dwt.__name__, args)

def awgn_dwt(img, mean, std, seed):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(mean):
        mean = np.full(4, mean)
    elif isinstance(mean, (list, np.ndarray)) and len(mean) == 4:
        mean = np.array(mean)
    if np.isscalar(std):
        std = np.full(4, std)
    elif isinstance(std, (list, np.ndarray)) and len(std) == 4:
        std = np.array(std)
    LL_noisy = awgn(LL, mean[0], std[0], seed)[0]
    LH_noisy = awgn(LH, mean[1], std[1], seed)[0]
    HL_noisy = awgn(HL, mean[2], std[2], seed)[0]
    HH_noisy = awgn(HH, mean[3], std[3], seed)[0]    
    coeffs_noisy = (LL_noisy, (LH_noisy, HL_noisy, HH_noisy))
    attacked = pywt.idwt2(coeffs_noisy, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, awgn_dwt.__name__, args)

def jpeg_dwt(img, qf):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(qf):
        qf = np.full(4, qf)
    elif isinstance(qf, (list, np.ndarray)) and len(qf) == 4:
        qf = np.array(qf)
    LL_compressed = jpeg_compression(LL, int(qf[0]))[0]
    LH_compressed = jpeg_compression(LH, int(qf[1]))[0]
    HL_compressed = jpeg_compression(HL, int(qf[2]))[0]
    HH_compressed = jpeg_compression(HH, int(qf[3]))[0]    
    coeffs_compressed = (LL_compressed, (LH_compressed, HL_compressed, HH_compressed))
    attacked = pywt.idwt2(coeffs_compressed, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, jpeg_dwt.__name__, args)

def resize_dwt(img, scale):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(scale):
        scale = np.full(4, scale)
    elif isinstance(scale, (list, np.ndarray)) and len(scale) == 4:
        scale = np.array(scale)
    LL_resized = resize(LL, scale[0])[0]
    LH_resized = resize(LH, scale[1])[0]
    HL_resized = resize(HL, scale[2])[0]
    HH_resized = resize(HH, scale[3])[0]    
    coeffs_resized = (LL_resized, (LH_resized, HL_resized, HH_resized))
    attacked = pywt.idwt2(coeffs_resized, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, resize_dwt.__name__, args)

# --------------------
#   Combo attacks
# --------------------

def resize_jpeg(img, qf, scale):
    args = {key: value for key, value in list(locals().items())[1:]}
    compressed = jpeg_compression(img, qf)[[0]]
    attacked = resize(compressed, scale)[0]
    return (attacked, resize_jpeg.__name__, args)

def blur_jpeg(img, qf, blur_func, sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    compressed = jpeg_compression(img, qf)[0]
    attacked = blur_func(compressed, sigma)[0]
    return (attacked, blur_jpeg.__name__, args)

def blur_awgn(img, mean, std, seed, blur_func, sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    noisy = awgn(img, mean, std, seed)[0]
    attacked = blur_func(noisy, sigma)[0]
    return (attacked, blur_awgn.__name__, args)

def jpeg_awgn(img, mean, std, seed, qf):
    args = {key: value for key, value in list(locals().items())[1:]}
    noisy = awgn(img, mean, std, seed)[0]
    attacked = jpeg_compression(noisy, qf)[0]
    return (attacked, jpeg_awgn.__name__, args)