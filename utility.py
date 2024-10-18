import numpy as np
from scipy.signal import convolve2d
from math import sqrt
import matplotlib.pyplot as plt

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