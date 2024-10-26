import numpy as np
from scipy.signal import convolve2d
from math import sqrt
import matplotlib.pyplot as plt
import cv2


def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0
  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  csf = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels


def visualize_images_with_desc(images, titles, figsize=(15, 6)):
    # Check if the number of images matches the number of titles
    if len(images) != len(titles):
        raise ValueError("The number of images must match the number of titles.")

    # Create a figure with the specified size
    plt.figure(figsize=figsize)

    # Loop through the images and titles to create subplots
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)  # Adjust the number of columns based on the number of images
        plt.title(title)
        plt.imshow(image, cmap='gray')
        plt.axis('off') 

    # Show the plot
    plt.tight_layout()  # Adjust the layout
    plt.show()

def create_perceptual_mask_2(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    mask = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def compute_brightness_sensitivity(subband):

    # Normalize brightness between 0 and 1
    min_brightness = np.min(subband)
    max_brightness = np.max(subband)
    # brightness_sensitivity = (subband - min_brightness) / (max_brightness - min_brightness + 1e-6)
    brightness_sensitivity = np.clip()
    
    # Invert to give higher sensitivity in dark areas (lower brightness = higher mask value)
    return 1 - brightness_sensitivity

def compute_edge_sensitivity(subband):

    # Compute image gradient (strong edges correspond to higher gradients)
    sobel_x = cv2.Sobel(subband, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(subband, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize gradient magnitude between 0 and 1
    # gradient_sensitivity = (gradient_magnitude - np.min(gradient_magnitude)) / (np.max(gradient_magnitude) - np.min(gradient_magnitude) + 1e-6)
    gradient_sensitivity = np.clip(gradient_magnitude, 0, 1)
    
    return gradient_sensitivity

def compute_texture_sensitivity(subband):
    
    # Compute local variance as a measure of texture
    mean = cv2.blur(subband, (3, 3))
    local_variance = cv2.blur((subband - mean) ** 2, (3, 3))
    
    # Normalize local variance between 0 and 1
    # texture_sensitivity = (local_variance - np.min(local_variance)) / (np.max(local_variance) - np.min(local_variance) + 1e-6)
    texture_sensitivity = np.clip(local_variance, 0, 1)
    
    return texture_sensitivity

def create_perceptual_mask_1(subband):

    mask = np.ones(subband.shape)
    mask += compute_brightness_sensitivity(subband) * compute_edge_sensitivity(subband) * compute_texture_sensitivity(subband)

    return mask

def modular_alpha(layer, theta, alpha):
    arrayLayer = [1.0, 0.32, 0.16, 0.1]
    arrayTheta = [1, sqrt(2), 1]

    return alpha * arrayLayer[layer] * arrayTheta[theta]

def get_locations(subband):
    sign = np.sign(subband)
    abs_subband = abs(subband)
    locations = np.argsort(-abs_subband, axis=None) # - sign is used to get descending order
    rows = subband.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates

    return abs_subband, sign, locations



def robustness_point(avg):
    if avg >= 53:
        points = 0
    elif avg >= 50:
        points = 1
    elif avg >= 47:
        points = 2
    elif avg >= 44:
        points = 3
    elif avg >= 41:
        points = 4
    elif avg >= 38:
        points = 5
    else:
        points = 6
    return points

def invisibility_point(avg):
    if avg >= 66:
        points = 6
    elif avg >= 62:
        points = 5
    elif avg >= 58:
        points = 4
    elif avg >= 54:
        points = 3
    elif avg >= 50:
        points = 4
    elif avg >= 35:
        points = 1
    else:
        points = 0
    return points


    
    
            
       
