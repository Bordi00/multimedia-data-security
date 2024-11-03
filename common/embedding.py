from utility import create_perceptual_mask_1, create_perceptual_mask_2, get_locations, modular_alpha
import pywt


def embed_watermark(subband, mark, layer, theta, alpha=0.5, mask_type=2, v='multiplicative'):

    if mask_type == 1:
        mask = create_perceptual_mask_1(subband)
    else:
        mask = create_perceptual_mask_2(subband)

    abs_subband, sign, locations = get_locations(subband) 

    watermarked = abs_subband.copy()
    for idx, (loc, mark_val) in enumerate(zip(locations[1:], mark)):
        x = locations[idx][0]
        y = locations[idx][1]
        if v == 'additive':
            watermarked[loc] += (modular_alpha(layer, theta, alpha) * mark_val * mask[x][y])
        elif v == 'multiplicative':
            watermarked[loc] *= 1 + (modular_alpha(layer, theta, alpha) * mark_val * mask[x][y])

    return sign * watermarked

def recursive_embedding(coeffs, mark, alpha, layer, max_layer, mask_type=2, v='multiplicative'):
    LL, (LH, HL, HH) = coeffs

    # Base case: if we reach layer 3, embed the watermark and return
    if layer == max_layer:
        watermarked_LH = embed_watermark(LH, mark, layer, 0, alpha, mask_type, v)
        watermarked_HL = embed_watermark(HL, mark, layer, 2, alpha, mask_type, v)
        watermarked_HH = embed_watermark(HH, mark, layer, 1, alpha, mask_type, v)

        watermarked_LL = pywt.idwt2((LL, (watermarked_LH, watermarked_HL, watermarked_HH)), 'haar')
        return watermarked_LL

    # Recursive case: perform another DWT and recurse
    coeffs_next = pywt.dwt2(LL, 'haar')
    watermarked_LL = recursive_embedding(coeffs_next, mark, alpha, layer + 1, max_layer, mask_type, v)

    # Embed the watermark at this layer
    watermarked_LH = embed_watermark(LH, mark, layer, 0, alpha, mask_type, v)
    watermarked_HL = embed_watermark(HL, mark, layer, 2, alpha, mask_type, v)
    watermarked_HH = embed_watermark(HH, mark, layer, 1, alpha, mask_type, v)

    # Return the inverse DWT of the watermarked image
    watermarked = pywt.idwt2((watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH)), 'haar')
    return watermarked


def embedding(image, mark, alpha, max_layer=2, mask_type=2, v='multiplicative'):
    # Initial wavelet decomposition
    coeffs = pywt.dwt2(image, 'haar')
    # Start recursive embedding from layer 0
    watermarked_image = recursive_embedding(coeffs, mark, alpha, layer=0, max_layer=max_layer, mask_type=mask_type, v=v)

    return watermarked_image