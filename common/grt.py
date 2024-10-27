# --------------------
#   Imports
# --------------------

import os                                           # standard
import sys
import random
import inspect
from math import sqrt                               # math and analysis
import numpy as np
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
#   Wrapper
# --------------------

attack_type = {
    'blur_gauss':       lambda img, **attack_args: blur_gauss(img, **attack_args),
    'blur_median':      lambda img, **attack_args: blur_median(img, **attack_args),
    'awgn':             lambda img, **attack_args: awgn(img, **attack_args),
    'jpeg_compression': lambda img, **attack_args: jpeg_compression(img, **attack_args),
    'resize':           lambda img, **attack_args: resize(img, **attack_args),
    'gauss_edge':       lambda img, **attack_args: gauss_edge(img, **attack_args),
    'median_edge':      lambda img, **attack_args: median_edge(img, **attack_args),
    'gauss_flat':       lambda img, **attack_args: gauss_flat(img, **attack_args), 
    'median_flat':      lambda img, **attack_args: median_flat(img, **attack_args),
    'awgn_edge':        lambda img, **attack_args: awgn_edge(img, **attack_args),
    'resize_jpeg':      lambda img, **attack_args: resize_jpeg(img, **attack_args),
    'gauss_jpeg':       lambda img, **attack_args: gauss_jpeg(img, **attack_args),
    'median_jpeg':      lambda img, **attack_args: median_jpeg(img, **attack_args),
    'gauss_awgn':       lambda img, **attack_args: gauss_awgn(img, **attack_args),
    'median_awgn':      lambda img, **attack_args: median_awgn(img, **attack_args),
    'jpeg_awgn':        lambda img, **attack_args: jpeg_awgn(img, **attack_args),
    'gauss_dwt':        lambda img, **attack_args: gauss_dwt(img, **attack_args),
    'median_dwt':       lambda img, **attack_args: median_dwt(img, **attack_args),
    'awgn_dwt':         lambda img, **attack_args: awgn_dwt(img, **attack_args),
    'jpeg_dwt':         lambda img, **attack_args: jpeg_dwt(img, **attack_args),
    'resize_dwt':       lambda img, **attack_args: resize_dwt(img, **attack_args),
    'gauss_edge_dwt':   lambda img, **attack_args: gauss_edge_dwt(img, **attack_args),
    'median_edge_dwt':  lambda img, **attack_args: median_edge_dwt(img, **attack_args),
    'gauss_flat_dwt':   lambda img, **attack_args: gauss_flat_dwt(img, **attack_args),
    'median_flat_dwt':  lambda img, **attack_args: median_flat_dwt(img, **attack_args),
    'awgn_edge_dwt':    lambda img, **attack_args: awgn_edge_dwt(img, **attack_args),
    'resize_jpeg_dwt':  lambda img, **attack_args: resize_jpeg_dwt(img, **attack_args),
    'gauss_jpeg_dwt':   lambda img, **attack_args: gauss_jpeg_dwt(img, **attack_args),
    'median_jpeg_dwt':  lambda img, **attack_args: median_jpeg_dwt(img, **attack_args),
    'gauss_awgn_dwt':   lambda img, **attack_args: gauss_awgn_dwt(img, **attack_args),
    'median_awgn_dwt':  lambda img, **attack_args: median_awgn_dwt(img, **attack_args),
    'jpeg_awgn_dwt':    lambda img, **attack_args: jpeg_awgn_dwt(img, **attack_args),
    'attack_edge_blur': lambda img, **attack_args: attack_edge_blur(img, **attack_args)
}

def attack(img_path, attack_id, attack_args): # TO DO change image into path
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return attack_type[attack_id](img, **attack_args)[0]       

# --------------------
#   Edge detection
# --------------------

def sobel_edge(img):
    sobelimage = img.copy()
    sobelx = cv2.Sobel(sobelimage, cv2.CV_64F, 1, 0, ksize = 3)
    sobely = cv2.Sobel(sobelimage, cv2.CV_64F, 0, 1, ksize = 3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = cv2.convertScaleAbs(magnitude)
    _, thresholded = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
    return thresholded
  
def canny_edge(img):
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
 #attack (img, "blur_gauss",{sigma:2})
def blur_gauss(img, sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    attacked = gaussian_filter(img, sigma)
    return (attacked, inspect.stack()[0][3], args)

def blur_median(img, size):
    args = {key: value for key, value in list(locals().items())[1:]}
    attacked = medfilt(img, size)
    return (attacked, inspect.stack()[0][3], args)

def awgn(img, mean, std, seed):
    args = {key: value for key, value in list(locals().items())[1:]}
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return (attacked, inspect.stack()[0][3], args)

def jpeg_compression(img, qf):
    args = {key: value for key, value in list(locals().items())[1:]}
    img = Image.fromarray(img)
    img = img.convert('L') 
    img.save('tmp.jpg', "JPEG", quality = qf)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype = np.uint8)
    os.remove('tmp.jpg')
    return (attacked, inspect.stack()[0][3], args)

def attack_edge_blur(img,attack_name, attack_args,sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    attacked_mask = attack_type[attack_name](img,**attack_args)[0]
    attacked_mask = blur_gauss(attacked_mask,sigma)[0]
    
    edges = canny_edge(img).astype(np.uint8) 
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    print(img.shape, edges.shape, attacked_mask.shape)
    img[edges > 0] = attacked_mask[edges > 0]
    return (img, inspect.stack()[0][3], args)

def resize(img, scale):
    args = {key: value for key, value in list(locals().items())[1:]}
    if scale == 1:
        return (img, resize.__name__, args)
    x, y = img.shape
    attacked = rescale(img, scale, anti_aliasing = True, mode = 'reflect')
    attacked = rescale(attacked, 1 / scale, anti_aliasing = True, mode = 'reflect')
    attacked = np.asarray(attacked * 255, dtype = np.uint8)
    attacked = cv2.resize(attacked, (y, x), interpolation = cv2.INTER_LINEAR)
    return (attacked, inspect.stack()[0][3], args)

# --------------------
#   Localized attacks
# --------------------

def gauss_edge(img, sigma, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    if edge_func == 0:
        edges = sobel_edge(img)
    elif edge_func == 1:
        edges = canny_edge(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    blurred_img = blur_gauss(img, sigma)[0]
    attacked = (1 - edges) * img + edges * blurred_img
    return (attacked, inspect.stack()[0][3], args)

def median_edge(img, size, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    if edge_func == 0:
        edges = sobel_edge(img)
    elif edge_func == 1:
        edges = canny_edge(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    blurred_img = blur_median(img, size)[0]
    attacked = (1 - edges) * img + edges * blurred_img
    return (attacked, inspect.stack()[0][3], args)

def gauss_flat(img, sigma, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    if edge_func == 0:
        edges = sobel_edge(img)
    elif edge_func == 1:
        edges = canny_edge(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    blurred_img = blur_gauss(img, sigma)[0]
    attacked = edges * img + (1 - edges) * blurred_img
    return (attacked, inspect.stack()[0][3], args)

def median_flat(img, size, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    if edge_func == 0:
        edges = sobel_edge(img)
    elif edge_func == 1:
        edges = canny_edge(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    blurred_img = blur_median(img, size)[0]
    attacked = edges * img + (1 - edges) * blurred_img
    return (attacked, inspect.stack()[0][3], args)

def awgn_edge(img, mean, std, seed, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    attacked = img.copy()
    edge_res = img.copy()
    global_awgn = awgn(img, mean, std, seed)[0]
    if edge_func == 0:
        edges = sobel_edge(img)
    elif edge_func == 1:
        edges = canny_edge(img)
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    edge_res[edges > 0] = [255]
    attacked[edges > 0] = global_awgn[edges > 0]
    return (attacked, inspect.stack()[0][3], args)

# --------------------
#   Combo attacks
# --------------------

def resize_jpeg(img, qf, scale):
    args = {key: value for key, value in list(locals().items())[1:]}
    compressed = jpeg_compression(img, qf)[0]
    attacked = resize(compressed, scale)[0]
    return (attacked, inspect.stack()[0][3], args)

def gauss_jpeg(img, qf, sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    compressed = jpeg_compression(img, qf)[0]
    attacked = blur_gauss(compressed, sigma)[0]
    return (attacked, inspect.stack()[0][3], args)

def median_jpeg(img, qf, size):
    args = {key: value for key, value in list(locals().items())[1:]}
    compressed = jpeg_compression(img, qf)[0]
    attacked = blur_median(compressed, size)[0]
    return (attacked, inspect.stack()[0][3], args)

def gauss_awgn(img, mean, std, seed, sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    noisy = awgn(img, mean, std, seed)[0]
    attacked = blur_gauss(noisy, sigma)[0]
    return (attacked, inspect.stack()[0][3], args)

def median_awgn(img, mean, std, seed, size):
    args = {key: value for key, value in list(locals().items())[1:]}
    noisy = awgn(img, mean, std, seed)[0]
    attacked = blur_median(noisy, size)[0]
    return (attacked, inspect.stack()[0][3], args)

def jpeg_awgn(img, mean, std, seed, qf):
    args = {key: value for key, value in list(locals().items())[1:]}
    noisy = awgn(img, mean, std, seed)[0]
    attacked = jpeg_compression(noisy, qf)[0]
    return (attacked, inspect.stack()[0][3], args)

# --------------------
#   Tranform attacks
# --------------------

def gauss_dwt(img, sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_blurred = blur_gauss(LL, sigma[0])[0]
    LH_blurred = blur_gauss(LH, sigma[1])[0]
    HL_blurred = blur_gauss(HL, sigma[2])[0]
    HH_blurred = blur_gauss(HH, sigma[3])[0]    
    coeffs_blurred = (LL_blurred, (LH_blurred, HL_blurred, HH_blurred))
    attacked = pywt.idwt2(coeffs_blurred, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def median_dwt(img, size):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_blurred = blur_median(LL, size[0])[0]
    LH_blurred = blur_median(LH, size[1])[0]
    HL_blurred = blur_median(HL, size[2])[0]
    HH_blurred = blur_median(HH, size[3])[0]    
    coeffs_blurred = (LL_blurred, (LH_blurred, HL_blurred, HH_blurred))
    attacked = pywt.idwt2(coeffs_blurred, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

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
    return (attacked, inspect.stack()[0][3], args)

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
    return (attacked, inspect.stack()[0][3], args)

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
    return (attacked, inspect.stack()[0][3], args)

def gauss_edge_dwt(img, sigma, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_blurred = gauss_edge(LL, sigma[0], edge_func)[0]
    LH_blurred = gauss_edge(LH, sigma[1], edge_func)[0]
    HL_blurred = gauss_edge(HL, sigma[2], edge_func)[0]
    HH_blurred = gauss_edge(HH, sigma[3], edge_func)[0]    
    coeffs_blurred = (LL_blurred, (LH_blurred, HL_blurred, HH_blurred))
    attacked = pywt.idwt2(coeffs_blurred, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def median_edge_dwt(img, size, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_blurred = median_edge(LL, size[0], edge_func)[0]
    LH_blurred = median_edge(LH, size[1], edge_func)[0]
    HL_blurred = median_edge(HL, size[2], edge_func)[0]
    HH_blurred = median_edge(HH, size[3], edge_func)[0]    
    coeffs_blurred = (LL_blurred, (LH_blurred, HL_blurred, HH_blurred))
    attacked = pywt.idwt2(coeffs_blurred, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def gauss_flat_dwt(img, sigma, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_blurred = gauss_flat(LL, sigma[0], edge_func)[0]
    LH_blurred = gauss_flat(LH, sigma[1], edge_func)[0]
    HL_blurred = gauss_flat(HL, sigma[2], edge_func)[0]
    HH_blurred = gauss_flat(HH, sigma[3], edge_func)[0]    
    coeffs_blurred = (LL_blurred, (LH_blurred, HL_blurred, HH_blurred))
    attacked = pywt.idwt2(coeffs_blurred, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def median_flat_dwt(img, size, edge_func):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_blurred = median_flat(LL, size[0], edge_func)[0]
    LH_blurred = median_flat(LH, size[1], edge_func)[0]
    HL_blurred = median_flat(HL, size[2], edge_func)[0]
    HH_blurred = median_flat(HH, size[3], edge_func)[0]    
    coeffs_blurred = (LL_blurred, (LH_blurred, HL_blurred, HH_blurred))
    attacked = pywt.idwt2(coeffs_blurred, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def awgn_edge_dwt(img, mean, std, seed, edge_func):
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
    LL_noisy = awgn_edge(LL, mean[0], std[0], seed, edge_func)[0]
    LH_noisy = awgn_edge(LH, mean[1], std[1], seed, edge_func)[0]
    HL_noisy = awgn_edge(HL, mean[2], std[2], seed, edge_func)[0]
    HH_noisy = awgn_edge(HH, mean[3], std[3], seed, edge_func)[0]    
    coeffs_noisy = (LL_noisy, (LH_noisy, HL_noisy, HH_noisy))
    attacked = pywt.idwt2(coeffs_noisy, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def resize_jpeg_dwt(img, qf, scale):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(qf):
        qf = np.full(4, qf)
    elif isinstance(qf, (list, np.ndarray)) and len(qf) == 4:
        qf = np.array(qf)
    if np.isscalar(scale):
        scale = np.full(4, scale)
    elif isinstance(scale, (list, np.ndarray)) and len(scale) == 4:
        scale = np.array(scale)
    LL_processed = resize_jpeg(LL, qf[0], scale[0])[0]
    LH_processed = resize_jpeg(LH, qf[1], scale[1])[0]
    HL_processed = resize_jpeg(HL, qf[2], scale[2])[0]
    HH_processed = resize_jpeg(HH, qf[3], scale[3])[0]    
    coeffs_processed = (LL_processed, (LH_processed, HL_processed, HH_processed))
    attacked = pywt.idwt2(coeffs_processed, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def gauss_jpeg_dwt(img, qf, sigma):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(qf):
        qf = np.full(4, qf)
    elif isinstance(qf, (list, np.ndarray)) and len(qf) == 4:
        qf = np.array(qf)
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_processed = gauss_jpeg(LL, qf[0], sigma[0])[0]
    LH_processed = gauss_jpeg(LH, qf[1], sigma[1])[0]
    HL_processed = gauss_jpeg(HL, qf[2], sigma[2])[0]
    HH_processed = gauss_jpeg(HH, qf[3], sigma[3])[0]    
    coeffs_processed = (LL_processed, (LH_processed, HL_processed, HH_processed))
    attacked = pywt.idwt2(coeffs_processed, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def median_jpeg_dwt(img, qf, size):
    args = {key: value for key, value in list(locals().items())[1:]}
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    if np.isscalar(qf):
        qf = np.full(4, qf)
    elif isinstance(qf, (list, np.ndarray)) and len(qf) == 4:
        qf = np.array(qf)
    if np.isscalar(size):
        size = np.full(4, size)
    elif isinstance(size, (list, np.ndarray)) and len(size) == 4:
        size = np.array(size)
    LL_processed = median_jpeg(LL, qf[0], size[0])[0]
    LH_processed = median_jpeg(LH, qf[1], size[1])[0]
    HL_processed = median_jpeg(HL, qf[2], size[2])[0]
    HH_processed = median_jpeg(HH, qf[3], size[3])[0]    
    coeffs_processed = (LL_processed, (LH_processed, HL_processed, HH_processed))
    attacked = pywt.idwt2(coeffs_processed, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def gauss_awgn_dwt(img, mean, std, seed, sigma):
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
    if np.isscalar(sigma):
        sigma = np.full(4, sigma)
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == 4:
        sigma = np.array(sigma)
    LL_proccessed = gauss_awgn(LL, mean[0], std[0], seed, sigma[0])[0]
    LH_proccessed = gauss_awgn(LH, mean[1], std[1], seed, sigma[1])[0]
    HL_proccessed = gauss_awgn(HL, mean[2], std[2], seed, sigma[2])[0]
    HH_proccessed = gauss_awgn(HH, mean[3], std[3], seed, sigma[3])[0]    
    coeffs_processed = (LL_proccessed, (LH_proccessed, HL_proccessed, HH_proccessed))
    attacked = pywt.idwt2(coeffs_processed, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def median_awgn_dwt(img, mean, std, seed, size):
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
    if np.isscalar(size):
        size = np.full(4, size)
    elif isinstance(size, (list, np.ndarray)) and len(size) == 4:
        size = np.array(size)
    LL_proccessed = median_awgn(LL, mean[0], std[0], seed, size[0])[0]
    LH_proccessed = median_awgn(LH, mean[1], std[1], seed, size[1])[0]
    HL_proccessed = median_awgn(HL, mean[2], std[2], seed, size[2])[0]
    HH_proccessed = median_awgn(HH, mean[3], std[3], seed, size[3])[0]    
    coeffs_processed = (LL_proccessed, (LH_proccessed, HL_proccessed, HH_proccessed))
    attacked = pywt.idwt2(coeffs_processed, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)

def jpeg_awgn_dwt(img, mean, std, seed, qf):
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
    if np.isscalar(qf):
        qf = np.full(4, qf)
    elif isinstance(qf, (list, np.ndarray)) and len(qf) == 4:
        qf = np.array(qf)
    LL_proccessed = jpeg_awgn(LL, mean[0], std[0], seed, qf[0])[0]
    LH_proccessed = jpeg_awgn(LH, mean[1], std[1], seed, qf[1])[0]
    HL_proccessed = jpeg_awgn(HL, mean[2], std[2], seed, qf[2])[0]
    HH_proccessed = jpeg_awgn(HH, mean[3], std[3], seed, qf[3])[0]    
    coeffs_processed = (LL_proccessed, (LH_proccessed, HL_proccessed, HH_proccessed))
    attacked = pywt.idwt2(coeffs_processed, 'haar')
    attacked = np.clip(attacked, 0, 255)
    attacked = attacked.astype(np.uint8)
    return (attacked, inspect.stack()[0][3], args)