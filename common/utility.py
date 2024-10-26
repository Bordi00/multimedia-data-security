import numpy as np
from scipy.signal import convolve2d
from math import sqrt
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import os
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

def create_perceptual_mask(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    mask = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
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




'''
total_history,total_succesfull_attacks,points =  utility.test_pipelines(
    alpha =0.48, 
    max_layer=1, 
    num_images=2,
    embedding_fn=embedding.embedding,
    attacks_list=attacks.attack_incremental_paramters,
    detection_fn=detection.detection)
'''
def test_pipelines(alpha,max_layer,num_images,embedding_fn,detection_fn,attacks_list):
    img_folder = 'sample_imgs'
    img_files =  [f for f in os.listdir(img_folder) if f.endswith(('.bmp'))]
    img_files = random.sample(img_files, num_images)
    images = []
    for file in img_files:
        img_path = os.path.join(img_folder, file)
        images.append(cv2.imread(img_path, 0))
    images = np.array(images) # optional
    visualize_images_with_desc(images, ['Original']*len(images))
    mark = np.load('ammhackati.npy')


    history = []    
    watermarked = []
    wpsnr_value = 0
    for img in images:
        watermarked.append(embedding_fn(img, mark, alpha,max_layer=max_layer))
    for i,img in enumerate(watermarked):
        wpsnr_value += wpsnr(img, images[i])
        
    print("meanw psnr after embedding ", wpsnr_value/len(watermarked))
    invisibility = invisibility_point(wpsnr_value/len(watermarked))
    total_history = []
    total_succesfull = []
    for i,wm in enumerate(watermarked):
        attack_functions = attacks_list
        progress_bar = tqdm(attack_functions, desc="Applying attacks")
        history = []
        succesfull_attack = []
        for attack_fn in progress_bar:
            param = attack_fn['start']
            detected = 1
            attack_name = ''
            while param <= attack_fn['end']:
                attacked,attack_name,usd = attack_fn['function'](wm, param)
                detected = detection_fn(images[i], wm,attacked, alpha, max_layer)
                wpsnr_attacked = wpsnr(wm, attacked)
                progress_bar.set_postfix({"image":i,"attack":attack_name , "wpsnr": wpsnr_attacked,"detected":detected,"param":usd})
                #utility.visualize_images_with_desc([images[i], wm, attacked], ['Original', 'Watermarked', 'Attacked'])
                param += attack_fn['increment_params']

                history.append({"images":i,"attack":attack_name , "wpsnr": wpsnr_attacked,"param":usd})
                if detected == 0:
                    succesfull_attack.append({"images":i,"attack":attack_name , "wpsnr": wpsnr_attacked,"param":usd})
                    break
        total_history.append(history)
        total_succesfull.append(succesfull_attack)
   
    max_attacked_wpsnr = 0
    mean_max_attacked_wpsnr = 0
    for succ in total_succesfull:
        for s in succ:
            #print("succesfull attack",s["attack"]," on image ", s['images'], " with wpsnr ", s['wpsnr'])
            max_attacked_wpsnr = max(max_attacked_wpsnr,s['wpsnr'])
        #print("max attacked wpsnr ", max_attacked_wpsnr, " on image ", s['images'])
        mean_max_attacked_wpsnr += max_attacked_wpsnr
    print("mean max attacked wpsnr ", mean_max_attacked_wpsnr/len(total_succesfull))
    robustenss = robustness_point(mean_max_attacked_wpsnr/len(total_succesfull))
    
    #print("estimate points for invisibility + robusteness ", points)
    
    # for hist in total_history:
    #     for h in hist:
    #         print("image ", h['images'], " wpsnr ", h['wpsnr'])
    # # for succ in succesfull_attack:
    #     succesfull_attacks.
    print("estimate points for invisibility + robusteness ", invisibility,robustenss," total points =",invisibility+robustenss)

    return total_history,total_succesfull,(invisibility,robustenss)
    
    
            
       
