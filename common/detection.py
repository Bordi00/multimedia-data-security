import numpy as np
import pywt
from utility import create_perceptual_mask, get_locations, modular_alpha, wpsnr


def extract_watermark(subband, watermarked_subband, layer, theta, alpha=0.5, v='multiplicative'):
    # Create perceptual mask for the subband
    mask = create_perceptual_mask(subband)
    abs_subband, sign, locations = get_locations(subband)
    abs_watermarked, _, _ = get_locations(watermarked_subband)
    mark_size = 1024

    extracted_mark = np.zeros(mark_size, dtype=np.float64)

    # Loop through each location (except the first one)
    for idx, loc in enumerate(locations[1:mark_size+1]):
        x = locations[idx][0]
        y = locations[idx][1]
        
        if v == 'additive':
            # Reverse the additive watermarking process to extract the mark
            extracted_mark[idx] = (watermarked_subband[loc] - subband[loc]) / (modular_alpha(layer, theta, alpha) * mask[x][y])
        elif v == 'multiplicative':
            # Reverse the multiplicative watermarking process to extract the mark
            extracted_mark[idx] = (watermarked_subband[loc] - subband[loc]) / (modular_alpha(layer, theta, alpha) * mask[x][y] * subband[loc])
        
    return np.clip(extracted_mark, 0, 1)

def detect_wm(image, watermarked, alpha, max_layer=2, v='multiplicative'):
    LL0_or, (LH0_or, HL0_or, HH0_or) = pywt.dwt2(image, 'haar')
    LL1_or, (LH1_or, HL1_or, HH1_or) = pywt.dwt2(LL0_or, 'haar')
    LL2_or, (LH2_or, HL2_or, HH2_or) = pywt.dwt2(LL1_or, 'haar')
     

    LL0_w, (LH0_w, HL0_w, HH0_w) = pywt.dwt2(watermarked, 'haar')
    LL1_w, (LH1_w, HL1_w, HH1_w) = pywt.dwt2(LL0_w, 'haar')
    LL2_w, (LH2_w, HL2_w, HH2_w) = pywt.dwt2(LL1_w, 'haar')
    
    extracted_wms = []
    w_ex = []
    if max_layer == 2:
        extracted_wms.append(extract_watermark(LH2_or, LH2_w, 2, 0, alpha=alpha, v=v))
        extracted_wms.append(extract_watermark(HL2_or, HL2_w, 2, 2, alpha=alpha, v=v))
        extracted_wms.append(extract_watermark(HH2_or, HH2_w, 2, 1, alpha=alpha, v=v))
        w_ex.append(sum(extracted_wms) / 3)

    extracted_wms = []
    if max_layer >= 1:
        extracted_wms.append(extract_watermark(LH1_or, LH1_w, 1, 0, alpha=alpha, v=v))
        extracted_wms.append(extract_watermark(HL1_or, HL1_w, 1, 2, alpha=alpha, v=v))
        extracted_wms.append(extract_watermark(HH1_or, HH1_w, 1, 1, alpha=alpha, v=v))
        w_ex.append(sum(extracted_wms) / 3)

    extracted_wms = []
    extracted_wms.append(extract_watermark(LH0_or, LH0_w, 0, 0, alpha=alpha, v=v))
    extracted_wms.append(extract_watermark(HL0_or, HL0_w, 0, 2, alpha=alpha, v=v))
    extracted_wms.append(extract_watermark(HH0_or, HH0_w, 0, 1, alpha=alpha, v=v))
    w_ex.append(sum(extracted_wms) / 3)


    return w_ex

def detection(original, watermarked, attacked, alpha, max_layer):
    ex_mark = detect_wm(original, watermarked, alpha, max_layer=max_layer)
    ex_attacked = detect_wm(original, attacked, alpha, max_layer=max_layer)
    thr = 0.7
    sim = []
    for w in ex_attacked:
        sim.append(similarity(ex_mark[0], w))

    sim = max(sim)
   
    if sim >= thr or wpsnr(watermarked, attacked) < 35:
        return 1
    return 0


def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0  # or return 0 or another appropriate value

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)
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


