
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from PIL import Image
import os
from utility import wpsnr
import cv2
from tqdm import tqdm

attack_list = [
    lambda img: awgn(img, std_range=(1,10), seed=123),                                     # AWGN attack
    lambda img: blur(img,sigma_range =(1, 5)),                                             # Blur attack
    lambda img: sharpening(img, sigma_range=(1,5), alpha_range=(1,5)),                     # Sharpening attack
    lambda img: median(img, kernel_size_w_range=[3,5,7], kernel_size_h_range=[3,5,7]),     # Median attack
    lambda img: resizing(img, scale_range = (0.3,1)),                                      # Resizing attack
    lambda img: jpeg_compression(img, QF_range=(1,100)),                                   #JPEG compression attack
    #lambda img: (img,'None','None')                                                       # No attack (identity)
]

"""
    Function to apply multiple attacks to an image and return the best attack, and the history of all attacks.
    img: input image
    attack_functions: list of attack functions
    times: number of times to apply each attack
"""
def multiple_attacks(img,attack_functions=attack_list,times=3): 
  # List of all attack functions
  
  max_wpsnr = -1
  history = {}
  best_attack = {}
  progress_bar = tqdm(attack_functions, desc="Applying attacks")
  for _, attack_fn in enumerate(progress_bar):
    for _ in range(times):
      (attacked,attack_name,used_params) = attack_fn(img)
      progress_bar.set_description(f"Applying {attack_name}")  
      attacked = attacked.astype(np.uint8)
      img = img.astype(np.uint8)
      psnr = cv2.PSNR(img, attacked)
    
      w = wpsnr(img, attacked) 

      if w > max_wpsnr:
        max_wpsnr = w
        best_attack ={
          'attacked': attacked,
          'attack_name': attack_name,
          'psnr': psnr,
          'wpsnr': w,
          'params': used_params}
      history.setdefault(attack_name, []).append({'psnr': psnr, 'wpsnr': w, 'params': used_params})

  print(f'Best attack: {best_attack["attack_name"]}, PSNR: {best_attack["psnr"]}, WPSNR: {best_attack["wpsnr"]}, Params: {best_attack["params"]}')
  return history, best_attack









##########################################   ATTACKS   ##########################################
def awgn(img, std_range, seed):
  mean = 0.0   # some constant
  std = random.randint(std_range[0], std_range[1])
  #np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  attacked = np.asarray(attacked,dtype=np.uint8)
  return (attacked, 'AWGN',"std: "+str(std))

def blur(img, sigma_range):
 
  sigma = random.randint(sigma_range[0], sigma_range[1])
  attacked = gaussian_filter(img, [sigma,sigma])
  return (attacked, 'Blur', "sigma: "+str(sigma))

def sharpening(img, sigma_range, alpha_range):

  sigma = random.randint(sigma_range[0], sigma_range[1])
  alpha = random.randint(alpha_range[0], alpha_range[1])
  #print(img/255)
  filter_blurred_f = gaussian_filter(img, sigma)

  attacked = img + alpha * (img - filter_blurred_f)
  return ( attacked, 'Sharpening', "sigma: "+str(sigma) + " alpha: "+str(alpha))

def median(img, kernel_size_w_range, kernel_size_h_range):

  k_w  = random.randint(0,len(kernel_size_w_range)-1)
  k_h  = random.randint(0,len(kernel_size_h_range)-1)
  kernel_size = [kernel_size_w_range[k_w], kernel_size_h_range[k_h]]
  attacked = medfilt(img, kernel_size)
  return (attacked, 'Median', "kernel_size: "+str(kernel_size))

def resizing(img, scale_range):
  
  x, y = img.shape
  scale = random.uniform(scale_range[0], scale_range[1])
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = np.asarray(attacked,dtype=np.uint8)
  attacked = cv2.resize(attacked, (y, x))
  attacked = attacked[:x, :y]

  return (attacked, 'Resizing', "scale: "+str(scale))

def jpeg_compression(img, QF_range):
  img = Image.fromarray(img)
  QF = random.randint(QF_range[0], QF_range[1])
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')

  return (attacked, 'JPEG Compression', "QF: "+str(QF))