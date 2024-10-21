
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




# Default List of attack functions
attack_list = [
    lambda img: awgn(img, std_range=(1,10), seed=123),                                     # AWGN attack
    lambda img: blur(img,sigma_range =(1, 5)),                                             # Blur attack
    lambda img: sharpening(img, sigma_range=(1,5), alpha_range=(1,5)),                     # Sharpening attack
    lambda img: median(img, kernel_size_w_range=[3,5,7], kernel_size_h_range=[3,5,7]),     # Median attack
    lambda img: resizing(img, scale_range = (0.3,1)),                                      # Resizing attack
    lambda img: jpeg_compression(img, QF_range=(1,100)),                                   #JPEG compression attack
    #lambda img: (img,'None','None')                                                       # No attack (identity)
]

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
      history.setdefault(attack_name, []).append({'psnr': psnr, 'wpsnr': w, 'params': used_params, 'attacked_image': attacked})

  print(f'Best attack: {best_attack["attack_name"]}, PSNR: {best_attack["psnr"]}, WPSNR: {best_attack["wpsnr"]}, Params: {best_attack["params"]}')
  return history, best_attack

def stats(history):
    stats = {}
    
    for attack_name, results in history.items():
        psnrs = [entry['psnr'] for entry in results]
        wpsnrs = [entry['wpsnr'] for entry in results]
        
        # Calculate mean PSNR and wPSNR
        mean_psnr = sum(psnrs) / len(psnrs)
        mean_wpsnr = sum(wpsnrs) / len(wpsnrs)
        
        # Get best and worst wPSNR
        best_wpsnr = max(wpsnrs)
        worst_wpsnr = min(wpsnrs)

        # Store the statistics for this attack
        stats[attack_name] = {
            'mean_psnr': mean_psnr,
            'mean_wpsnr': mean_wpsnr,
            'best_wpsnr': best_wpsnr,
            'worst_wpsnr': worst_wpsnr
        }
    mean_psnr = sum([s['mean_psnr'] for s in stats.values()]) / len(stats)
    mean_wpsnr = sum([s['mean_wpsnr'] for s in stats.values()]) / len(stats)
    best_wpsnr = max([s['best_wpsnr'] for s in stats.values()])
    worst_wpsnr = min([s['worst_wpsnr'] for s in stats.values()])
      
    stats["overall"] ={
            'mean_psnr': mean_psnr,
            'mean_wpsnr': mean_wpsnr,
            'best_wpsnr': best_wpsnr,
            'worst_wpsnr': worst_wpsnr
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