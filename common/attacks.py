
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from PIL import Image
import os
import pywt
from utility import wpsnr
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt




''' EXAMPLE USAGE
import sys
sys.path.append(os.path.join(os.getcwd(), 'common'))
import importlib
import attacks as attacks

# Reload the entire module, not the function
importlib.reload(attacks)

(history,_) = attacks.multiple_attacks(wm)   #INSERT YOUR EMBEDDED IMAGE HERE
#print(history)
stats = attacks.stats(history)               #compute the statistics
print(stats['overall'])
attacks.plot_stats(stats)                    #visualize the statics
'''



prova = [
    lambda img: apply_median_to_dwt(img,kernel_size_w=random.choice([3,5,7,9,11]), kernel_size_h=random.choice([3,5,7,9,11]))  
]

attack_list = [
    lambda img: awgn(img, std=random.randint(10,100), seed=123),                                            # AWGN attack
    lambda img: blur(img,sigma=random.randint(1, 40)),                                                      # Blur attack
    lambda img: sharpening(img, sigma=random.randint(1,30), alpha=random.randint(1,30)),                    # Sharpening attack
    lambda img: median(img, kernel_size_w=random.choice([3,5,7,9,11]), kernel_size_h=random.choice([3,5,7,9,11])),     # Median attack
    lambda img: resizing(img, scale = random.uniform(0.3,1)),                                      # Resizing attack
    lambda img: jpeg_compression(img, QF=random.randint(1,100)),                                   #JPEG compression attack
    lambda img: apply_gaussian_blur_to_dwt(img,random.randint(1,20) )  ,
    lambda img: apply_median_to_dwt(img,kernel_size_w=random.choice([3,5,7,9,11]), kernel_size_h=random.choice([3,5,7,9,11]))
    #lambda img: (img,'None','None')                                                       # No attack (identity)
]

attack_list2 = [
    lambda img,p: awgn(img, std=p, seed=123),                                            # AWGN attack
    lambda img,p: blur(img,sigma=p),                                                      # Blur attack
    lambda img,p: sharpening(img, sigma=p, alpha=p),                    # Sharpening attack
    lambda img,p: median(img, kernel_size_w=p, kernel_size_h=p),     # Median attack
    lambda img,p: resizing(img, scale = p),                                      # Resizing attack
    lambda img,p: jpeg_compression(img, QF=p),                                   #JPEG compression attack
    lambda img,p: apply_gaussian_blur_to_dwt(img,p )  ,
    lambda img,p: apply_median_to_dwt(img,kernel_size_w=p, kernel_size_h=p)
    #lambda img: (img,'None','None')                                                       # No attack (identity)
]

attack_incremental_paramters = [
  {
    'start':20,
    'increment_params' : 5 , 
    'end': 100,
    'function': lambda img,p: awgn(img, std=p, seed=123),  
  },
  {
      'start':0,
      'increment_params' : 1 ,  
      'end': 5,
      'function': lambda img,p: blur(img,sigma=p), 
  },
 {
      'start':0,
      'increment_params' : 0.5 ,  
      'end': 5,
      'function': lambda img,p: sharpening(img, sigma=p, alpha=p), 
      },
   {
      'start':3,
      'increment_params' : 2 ,  
      'end': 100,
      'function': lambda img,p: median(img, kernel_size_w=p, kernel_size_h=p), 
      },
   {
      'start':0.3,
      'increment_params' : 0.1 ,  
      'end': 1,
      'function': lambda img,p: resizing(img, scale = p),  
      },
  {
      'start':0,
      'increment_params' : 5 , 
      'end': 100 ,
      'function': lambda img,p: jpeg_compression(img, QF= (100-p)),  
      },
  {
      'start':1,
      'increment_params' : 5 ,  
      'end': 30,
      'function': lambda img,p: apply_gaussian_blur_to_dwt(img,p ) ,   
      },
  {
      'start':3,
      'increment_params' : 2 , 
      'end': 11,
      'function': lambda img,p: apply_median_to_dwt(img,kernel_size_w=p, kernel_size_h=p),    
      }
    

]

# Default List of attack functions
def random_attack_list():
  return attack_list


def mergeMultipleHistory(first,second):
  for attack_name, results in first.items():
    second[attack_name].extend(results)
  return second

"""
    Function to apply multiple attacks to an image and return the best attack, and the history of all attacks.
    img: input image
    attack_functions: list of attack functions
    times: number of times to apply each attack
"""
def multiple_attacks(img,attack_functions=attack_list,times=3,alpha=0.8,mark=None): 
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
      
      #Detect the watermark
      # w_ex = detection(img, attacked, alpha)
      
      # # for w  un

      # #Compute the similarity and threshold
      # sim = similarity(mark, w_ex)
      # T = 0.7#compute_thr(sim, N, mark)
      # lost = False
      # if sim > T:
      #     print('Mark has been found. SIM = %f' % sim)
      #     lost = False
      # else:
      #     print('Mark has been lost. SIM = %f' % sim)
      #     lost = True
          
      history.setdefault(attack_name, []).append({'psnr': psnr, 'wpsnr': w, 'params': used_params, 'attacked_image': attacked})#'lost': lost,'similarity': sim

  print(f'Best attack: {best_attack["attack_name"]}, PSNR: {best_attack["psnr"]}, WPSNR: {best_attack["wpsnr"]}, Params: {best_attack["params"]}')
  return history, best_attack

def stats(history):
    stats = {}
    lost= 0
    total_lost = 0
    for attack_name, results in history.items():
        psnrs = [entry['psnr'] for entry in results]
        wpsnrs = [entry['wpsnr'] for entry in results]
        
        # Calculate mean PSNR and wPSNR
        mean_psnr = sum(psnrs) / len(psnrs)
        mean_wpsnr = sum(wpsnrs) / len(wpsnrs)
        
        # Get best and worst wPSNR
        best_wpsnr = max(wpsnrs)
        worst_wpsnr = min(wpsnrs)
        #lost += sum([entry['lost'] for entry in results])
        # Store the statistics for this attack
        stats[attack_name] = {
            'mean_psnr': mean_psnr,
            'mean_wpsnr': mean_wpsnr,
            'best_wpsnr': best_wpsnr,
            'worst_wpsnr': worst_wpsnr,
            #'lost': lost
        }
        # total_lost += lost
        # lost = 0
    mean_psnr = sum([s['mean_psnr'] for s in stats.values()]) / len(stats)
    mean_wpsnr = sum([s['mean_wpsnr'] for s in stats.values()]) / len(stats)
    best_wpsnr = max([s['best_wpsnr'] for s in stats.values()])
    worst_wpsnr = min([s['worst_wpsnr'] for s in stats.values()])
    #total_lost = total_lost
      
    stats["overall"] ={
            'mean_psnr': mean_psnr,
            'mean_wpsnr': mean_wpsnr,
            'best_wpsnr': best_wpsnr,
            'worst_wpsnr': worst_wpsnr,
            'total_lost': total_lost
        }

      
    
    return stats

def plot_stats(stats):
    attack_names = list(stats.keys())
    mean_psnrs = [s['mean_psnr'] for s in stats.values()]
    mean_wpsnrs = [s['mean_wpsnr'] for s in stats.values()]
    best_wpsnrs = [s['best_wpsnr'] for s in stats.values()]
    worst_wpsnrs = [s['worst_wpsnr'] for s in stats.values()]

    x = range(len(attack_names))  # X-axis locations

    # Set up the bar plot
    plt.figure(figsize=(15, 8))

    # Plot mean PSNR
    plt.subplot(2, 2, 1)
    plt.bar(x, mean_psnrs, color='blue')
    plt.xticks(x, attack_names, rotation=45)
    plt.ylabel('Mean PSNR (dB)')
    plt.title('Mean PSNR for Each Attack')

    # Plot mean wPSNR
    plt.subplot(2, 2, 2)
    plt.bar(x, mean_wpsnrs, color='orange')
    plt.xticks(x, attack_names, rotation=45)
    plt.ylabel('Mean wPSNR (dB)')
    plt.title('Mean wPSNR for Each Attack')

    # Plot best wPSNR
    plt.subplot(2, 2, 3)
    plt.bar(x, best_wpsnrs, color='green')
    plt.xticks(x, attack_names, rotation=45)
    plt.ylabel('Best wPSNR (dB)')
    plt.title('Best wPSNR for Each Attack')

    # Plot worst wPSNR
    plt.subplot(2, 2, 4)
    plt.bar(x, worst_wpsnrs, color='red')
    plt.xticks(x, attack_names, rotation=45)
    plt.ylabel('Worst wPSNR (dB)')
    plt.title('Worst wPSNR for Each Attack')

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()



##########################################   ATTACKS   ##########################################

# attacked_image = attacks("path_to_image.jpg", "AWGN", [std])
# attacked_image = attacks("path_to_image.jpg", "Blur", [sigma])
# attacked_image = attacks("path_to_image.jpg", "Sharpening", [sigma, alpha])
# attacked_image = attacks("path_to_image.jpg", "Median", [kernel_size_w, kernel_size_h])
# attacked_image = attacks("path_to_image.jpg", "Resizing", [scale])
# attacked_image = attacks("path_to_image.jpg", "JPEG Compression", [QF])

def attacks(image_path, attack_name, param_array):

  img = cv2.imread(image_path, 0)
  if attack_name == 'AWGN':
    return awgn(img, param_array[0], seed=123)[0]
  elif attack_name == 'Blur':
    return blur(img, param_array[0])[0]
  elif attack_name == 'Sharpening':
    return sharpening(img, param_array[0], param_array[1])[0]
  elif attack_name == 'Median':
    return median(img, param_array[0], param_array[1])[0]
  elif attack_name == 'Resizing':
    return resizing(img, param_array[0])[0]
  elif attack_name == 'JPEG Compression':
    return jpeg_compression(img, param_array[0])[0]
  else:
    return img



def awgn(img, std, seed):
  mean = 0.0   # some constant
  std = std
  #np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  attacked = np.asarray(attacked,dtype=np.uint8)
  return (attacked, 'AWGN',"std: "+str(std))

def blur(img, sigma):
 
  sigma = sigma
  attacked = gaussian_filter(img, [sigma,sigma])
  return (attacked, 'Blur', "sigma: "+str(sigma))

def sharpening(img, sigma, alpha):

  sigma = sigma
  alpha = alpha
  #print(img/255)
  filter_blurred_f = gaussian_filter(img, sigma)

  attacked = img + alpha * (img - filter_blurred_f)
  return ( attacked, 'Sharpening', "sigma: "+str(sigma) + " alpha: "+str(alpha))

def median(img, kernel_size_w, kernel_size_h):

  kernel_size = [kernel_size_w, kernel_size_h]
  attacked = medfilt(img, kernel_size)
  return (attacked, 'Median', "kernel_size: "+str(kernel_size))

def resizing(img, scale):
  x, y = img.shape
  attacked = rescale(img, scale, anti_aliasing=True, mode='reflect')
  attacked = rescale(attacked, 1/scale, anti_aliasing=True, mode='reflect')
  attacked = np.asarray(attacked * 255, dtype=np.uint8)
  attacked = cv2.resize(attacked, (y, x), interpolation=cv2.INTER_LINEAR)

  return (attacked, 'Resizing', "scale: "+str(scale))

def jpeg_compression(img, QF):
  img = Image.fromarray(img)
  QF = QF
  img = img.convert('L')
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')

  return (attacked, 'JPEG Compression', "QF: "+str(QF))


def apply_gaussian_blur_to_dwt(image, sigma=1):
    # Step 1: Perform 2D DWT
    coeffs2 = pywt.dwt2(image, 'haar')  # Using Haar wavelet for simplicity
    LL, (LH, HL, HH) = coeffs2

    # Step 2: Apply Gaussian blur to LH, HL, and HH components
    LH_blurred = gaussian_filter(LH, sigma=sigma)
    HL_blurred = gaussian_filter(HL, sigma=sigma)
    HH_blurred = gaussian_filter(HH, sigma=sigma)

    # Step 3: Reconstruct the image using the blurred coefficients
    coeffs2_blurred = (LL, (LH_blurred, HL_blurred, HH_blurred))
    reconstructed_image = pywt.idwt2(coeffs2_blurred, 'haar')

    # Ensure pixel values are within valid range
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    return (reconstructed_image.astype(np.uint8),'DWT_GAUSS', "sigma: "+str(sigma))

def apply_gaussian_blur_to_dwt2(image, sigma=1):
    # Step 1: Perform 2D DWT
    coeffs2 = pywt.dwt2(image, 'haar')  # Using Haar wavelet for simplicity
    LL, (LH, HL, HH) = coeffs2

    # Step 2: Apply Gaussian blur to LH, HL, and HH components
    HL_blurred = gaussian_filter(HL, sigma=sigma)
    HH_blurred = gaussian_filter(HH, sigma=sigma)

    # Step 3: Reconstruct the image using the blurred coefficients
    coeffs2_blurred = (LL, (LH, HL_blurred, HH_blurred))
    reconstructed_image = pywt.idwt2(coeffs2_blurred, 'haar')

    # Ensure pixel values are within valid range
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    return (reconstructed_image.astype(np.uint8),'DWT_GAUSS2', "sigma: "+str(sigma))


def apply_median_to_dwt(image, kernel_size_w, kernel_size_h):
    # Step 1: Perform 2D DWT
    coeffs2 = pywt.dwt2(image, 'haar')  # Using Haar wavelet for simplicity
    LL, (LH, HL, HH) = coeffs2

    # Step 2: Apply Gaussian blur to LH, HL, and HH components
    LH_blurred = medfilt(LH, [kernel_size_w, kernel_size_h])
    HL_blurred = medfilt(HL, [kernel_size_w, kernel_size_h])
    HH_blurred = medfilt(HH, [kernel_size_w, kernel_size_h])

    # Step 3: Reconstruct the image using the blurred coefficients
    coeffs2_blurred = (LL, (LH_blurred, HL_blurred, HH_blurred))
    reconstructed_image = pywt.idwt2(coeffs2_blurred, 'haar')

    # Ensure pixel values are within valid range
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    return (reconstructed_image.astype(np.uint8),'DWT_MEIDAN',"kernel_size: "+str([kernel_size_w,kernel_size_h]))