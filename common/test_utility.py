import embedding
import detection
import attacks
from utility import visualize_images_with_desc, invisibility_point, wpsnr, robustness_point
from tqdm import tqdm
import random
import os
import cv2
import numpy as np



def test_pipelines(alpha,max_layer,num_images,embedding_fn=embedding.embedding,detection_fn=detection.detection,attacks_list=attacks.attack_incremental_paramters,embedding_mask=2,wm_list = None,img_list_path = None):

    '''
    total_history,total_succesfull_attacks,points =  utility.test_pipelines(
        alpha =0.48, 
        max_layer=1, 
        num_images=2,
        embedding_fn=embedding.embedding,
        attacks_list=attacks.attack_incremental_paramters,
        detection_fn=detection.detection)
    '''
    images = []
    if img_list_path is  None:
        img_folder = 'challenge_imgs'
        img_files =  [f for f in os.listdir(img_folder) if f.endswith(('.bmp'))]
        img_files = random.sample(img_files, num_images)
        images = []
        for file in img_files:
            img_path = os.path.join(img_folder, file)
            images.append(cv2.imread(img_path, 0))
        images = np.array(images) # optional
        visualize_images_with_desc(images, ['Original']*len(images))
        mark = np.load('ammhackati.npy')
    else:
        for path in img_list_path:
            images.append(cv2.imread(path)) 
        
    if wm_list is None:
        history = []  
        wpsn_show = [] 
        watermarked = []
        wpsnr_value = 0
        for img in images:
            watermarked.append(embedding_fn(img, mark, alpha,max_layer=max_layer,mask_type=embedding_mask))
        for i,img in enumerate(watermarked):
            wpsn_show.append(wpsnr(img, images[i]))
            wpsnr_value += wpsnr(img, images[i])
        visualize_images_with_desc(watermarked, wpsn_show)
        print("meanw psnr after embedding ", wpsnr_value/len(watermarked))
        invisibility = invisibility_point(wpsnr_value/len(watermarked))
    else:
        watermarked = wm_list
        for path in img_list_path:
            images.append(cv2.imread(path)) 
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
                detected,q = detection_fn(images[i], wm,attacked, alpha, max_layer)
                wpsnr_attacked = q
                progress_bar.set_postfix({"image":i,"attack":attack_name , "wpsnr": wpsnr_attacked,"detected":detected,"param":usd})
                #visualize_images_with_desc([images[i], wm, attacked], ['Original', 'Watermarked', 'Attacked'])
                param += attack_fn['increment_params']

                history.append({"images":i,"attack":attack_name , "wpsnr": wpsnr_attacked,"param":usd})
                if detected == 0:
                    succesfull_attack.append({"images":i,"attack":attack_name , "wpsnr": wpsnr_attacked,"param":usd})
                    break
        total_history.append(history)
        total_succesfull.append(succesfull_attack)
   
    mean_max_attacked_wpsnr = 0
    for succ in total_succesfull:
        max_attacked_wpsnr = 0
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




def watermarked_images(alpha,max_layer,num_images,embedding_fn=embedding.embedding,embedding_mask=2,img_folder="challenge_imgs"):

    '''
    total_history,total_succesfull_attacks,points =  utility.test_pipelines(
        alpha =0.48, 
        max_layer=1, 
        num_images=2,
        embedding_fn=embedding.embedding,
        attacks_list=attacks.attack_incremental_paramters,
        detection_fn=detection.detection)
    '''
    img_folder = img_folder
    img_files =  [f for f in os.listdir(img_folder) if f.endswith(('.bmp'))]
    img_files = random.sample(img_files, num_images)
    images = []
    for file in img_files:
        img_path = os.path.join(img_folder, file)
        images.append(cv2.imread(img_path, 0))
    images = np.array(images) # optional
    visualize_images_with_desc(images, ['Original']*len(images))
    mark = np.load('ammhackati.npy')


    history_w = []    
    watermarked = []
    wpsnr_value = 0
    for img in images:
        watermarked.append(embedding_fn(img, mark, alpha,max_layer=max_layer,mask_type=embedding_mask))
    for i,img in enumerate(watermarked):
        history_w.append(wpsnr(img, images[i]))
        wpsnr_value += wpsnr(img, images[i])
    visualize_images_with_desc(watermarked, history_w)
        
    print("meanw psnr after embedding ", wpsnr_value/len(watermarked))
    invisibility = invisibility_point(wpsnr_value/len(watermarked))
    print("estimate points for invisibility ", invisibility)
    return watermarked,images,invisibility