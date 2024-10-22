import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt

import os
from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from math import sqrt

def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0
  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  w = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(w,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels


def awgn(img, std, seed):
  mean = 0.0   # some constant
  #np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  return attacked


def embeddingwt(image, mark, alpha=0.1, v='multiplicative'):

    # Generate a watermark
    # mark = np.random.uniform(0.0, 1.0, mark_size)
    # mark = np.uint8(np.rint(mark))
    # np.save('mark.npy', mark)

    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2

    coeffs3 = pywt.dwt2(LL, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs3

    # Get the locations in LH
    sign_LH2 = np.sign(LH2)
    abs_LH2 = abs(LH2)
    locations_LH2 = np.argsort(-abs_LH2,axis=None) # - sign is used to get descending order
    rows_LH2 = LH2.shape[0]
    locations_LH2 = [(val//rows_LH2, val%rows_LH2) for val in locations_LH2] # locations as (x,y) coordinates

    # Get the locations in HL
    sign_HL2 = np.sign(HL2)
    abs_HL2 = abs(HL2)
    locations_HL2 = np.argsort(-abs_HL2,axis=None) # - sign is used to get descending order
    rows_HL2 = HL2.shape[0]
    locations_HL2 = [(val//rows_HL2, val%rows_HL2) for val in locations_HL2] # locations as (x,y) coordinates

    # Get the locations in HH
    sign_HH2 = np.sign(HH2)
    abs_HH2 = abs(HH2)
    locations_HH2 = np.argsort(-abs_HH2,axis=None) # - sign is used to get descending order
    rows_HH2 = HH2.shape[0]
    locations_HH2 = [(val//rows_HH2, val%rows_HH2) for val in locations_HH2] # locations as (x,y) coordinates
    # Embed the watermark in LH
    watermarked_LH2 = abs_LH2.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations_LH2[1:], mark)):
        if v == 'additive':
            watermarked_LH2[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_LH2[loc] *= 1 + ( alpha * mark_val)
    
     # Embed the watermark in LH
    watermarked_HL2 = abs_HL2.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations_HL2[1:], mark)):
        if v == 'additive':
            watermarked_HL2[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_HL2[loc] *= 1 + ( alpha * mark_val)
    
     # Embed the watermark in HH
    watermarked_HH2 = abs_HH2.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations_HH2[1:], mark)):
        if v == 'additive':
            watermarked_HH2[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_HH2[loc] *= 1 + ( alpha * mark_val)

    # Restore sign and o back to spatial domain
    watermarked_LH2 *= sign_LH2
    watermarked_HL2 *= sign_HL2
    watermarked_HH2 *= sign_HH2
    watermarked_LL = pywt.idwt2((LL2, (watermarked_LH2, watermarked_HL2, watermarked_HH2)), 'haar')
    watermarked = pywt.idwt2((watermarked_LL, (LH, HL, HH)), 'haar')

    return watermarked



def detectwt(image, watermarked, alpha=0.5, mark_size=1024, v='multiplicative'):
    #ori_dct = dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')
    coeffs2 = pywt.dwt2(image, 'haar')
    LL_or, (LH_or, HL_or, HH_or) = coeffs2
    #wat_dct = dct(dct(watermarked,axis=0, norm='ortho'),axis=1, norm='ortho')
    coeffs2 = pywt.dwt2(watermarked, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs2

    #ori_dct = dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')
    coeffs3 = pywt.dwt2(LL_or, 'haar')
    LL2_or, (LH2_or, HL2_or, HH2_or) = coeffs3
    #wat_dct = dct(dct(watermarked,axis=0, norm='ortho'),axis=1, norm='ortho')
    coeffs3 = pywt.dwt2(LL_w, 'haar')
    LL2_w, (LH2_w, HL2_w, HH2_w) = coeffs3

    # Get the locations in LH
    sign_LH2 = np.sign(LH2_or)
    abs_LH2 = abs(LH2_or)
    locations_LH2 = np.argsort(-abs_LH2,axis=None) # - sign is used to get descending order
    rows_LH2 = LH2_or.shape[0]
    locations_LH2 = [(val//rows_LH2, val%rows_LH2) for val in locations_LH2] # locations as (x,y) coordinates

    # Get the locations in HL
    sign_HL2 = np.sign(HL2_or)
    abs_HL2 = abs(HL2_or)
    locations_HL2 = np.argsort(-abs_HL2,axis=None) # - sign is used to get descending order
    rows_HL2 = HL2_or.shape[0]
    locations_HL2 = [(val//rows_HL2, val%rows_HL2) for val in locations_HL2] # locations as (x,y) coordinates

    # Get the locations in HH
    sign_HH2 = np.sign(HH2_or)
    abs_HH2 = abs(HH2_or)
    locations_HH2 = np.argsort(-abs_HH2,axis=None) # - sign is used to get descending order
    rows_HH2 = HH2_or.shape[0]
    locations_HH2 = [(val//rows_HH2, val%rows_HH2) for val in locations_HH2] # locations as (x,y) coordinates

    # Generate a watermark
    w_ex1 = np.zeros(mark_size, dtype=np.float64)
    w_ex2 = np.zeros(mark_size, dtype=np.float64)
    w_ex3 = np.zeros(mark_size, dtype=np.float64)

    # Embed the watermark
    for idx, loc in enumerate(locations_LH2[1:mark_size+1]):
        if v=='additive':
            w_ex1[idx] =  (LH2_w[loc] - LH2_or[loc]) /alpha
        elif v=='multiplicative':
            w_ex1[idx] =  (LH2_w[loc] - LH2_or[loc]) / (alpha*LH2_or[loc])
    
     # Embed the watermark
    for idx, loc in enumerate(locations_HL2[1:mark_size+1]):
        if v=='additive':
            w_ex2[idx] =  (HL2_w[loc] - HL2_or[loc]) /alpha
        elif v=='multiplicative':
            w_ex2[idx] =  (HL2_w[loc] - HL2_or[loc]) / (alpha*HL2_or[loc])

     # Embed the watermark
    for idx, loc in enumerate(locations_HH2[1:mark_size+1]):
        if v=='additive':
            w_ex3[idx] =  (HH2_w[loc] - HH2_or[loc]) /alpha
        elif v=='multiplicative':
            w_ex3[idx] =  (HH2_w[loc] - HH2_or[loc]) / (alpha*HH2_or[loc])
    
    w_ex = (w_ex1 + w_ex2 + w_ex3)/3
    return w_ex


def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def compute_thr(sim, mark, mark_size=1024, N=1000):
    SIM = np.zeros(N)
    for i in range(N):
        r = np.random.uniform(0.0, 1.0, mark_size)
        SIM[i] = (similarity(mark, r))
    SIMs = SIM.copy()
    SIM.sort()
    t = SIM[-1]
    T = t + (0.1*t)
    print('Threshold: ', T)
    return T, SIMs

    print('Threshold: ', T)
    return T, similiarities

imgs = [cv2.imread(os.path.join('./sample_imgs', img), 0) for img in os.listdir('./sample_imgs')]
img_path='lena_grey.bmp'
N = 1024
mark = np.load('ammhackati.npy')

#v = 'additive'
#alpha = 24 
v = 'multiplicative'
np.random.seed(seed=123)

wm_imgs = []
wpsnr_wm_imgs = []

for img in imgs:
    wm = embeddingwt(img, mark, 0.5)
    wm_imgs.append(wm)
    wpsnr_wm_imgs.append(wpsnr(img, wm))

for org_img, wm_img, quality in zip(imgs, wm_imgs, wpsnr_wm_imgs):
    w_ex = detectwt(org_img, wm_img, 0.5)
    attacked_wimg = awgn(wm_img, 24, np.random.seed(123))
    w_ex_attacked = detectwt(org_img, attacked_wimg, 0.5)

    print("WPSNR watermarked img before being attacked: ", quality);
    print("WPSNR watermarked img after being attackted: ", wpsnr(attacked_wimg, org_img))

    sim_not_attacked = similarity(mark, w_ex)
    sim_attacked = similarity(w_ex, w_ex_attacked)

    t, _ = compute_thr(sim_not_attacked, mark)

    print('\n')
    if sim_not_attacked > t:
        print('Mark has been found in the non-attacked image. SIM = %f' % sim_not_attacked)
    else:
        print('Mark has been lost in the non-attacked image. SIM = %f' % sim_not_attacked)
    if sim_attacked > t:
        print('Mark has been found in the attacked image. SIM = %f' % sim_attacked)
    else:
        print('Mark has been lost in the attacked image. SIM = %f' % sim_attacked)
    print('--------------------------')
    print('\n')



    