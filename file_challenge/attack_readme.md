# Attack File
The attack file contains different types of attack, that can be invoked through the `attack` wrapper function:

- Inputs:
    - `image_path`: path to the image to attack
    - `attack_name`: name of the attack function to apply
    - `args`: dictionary of argument specific to the selected attack_name
- Output:
    - `img`: the resulting image after the attack has been applied

## Features
- **Global Attacks**: Affect the entire image. Available options include Gaussian blur, median blur, additive white Gaussian noise (AWGN), JPEG compression, and sharpening.
- **Localized Attacks**: Target specific regions within the image, such as edges, by applying transformations selectively based on edge-detection methods.
- **Combination Attacks**: Chain multiple transformations in sequence for more complex effects, such as resizing + JPEG compression.
- **Wavelet Transform Attacks**: Use Discrete Wavelet Transform (DWT) to apply attacks at different frequency levels.

## Example
Here some example of usage
### Jpeg compression
The following code snippet applies jpeg compression to the image `watermarked_0.bmp` with a quality factor of 10.
```
attacked = attack.attack('watermarked_0.bmp',"jpeg_compression", {'qf':10})
```

### Blur edge
The following code snippet applies a blurring only on the edges of the image, in order to increase the wpsnr. The edge_func specifies the function used to detect the edge: 0 for Sobel, 1 for Canny.
```
attacked = attack.attack('watermarked_0.bmp',"gauss_edge", {'sigma':0.4, 'edge_func':0})
```

### Sharp + blur
The following code snippet performs a global blur of the image and subsequently apply sharpening, in order to increase the wpsnr.
```
attacked = attack.attack('watermarked_0.bmp',"sharp_gauss",{'sigma':0.5, 'alpha':0.5})
```

### Awgn to dwt
The following code snippet applies AWGN to the Discrete Wavelet Transform (DWT). Parameters can be provided as either scalars or arrays. To apply the same effect uniformly across all DWT subbands scalars are used. However, if you want to specify different values for each subband array should be used.
```
attacked = attack.attack('watermarked_0.bmp',"awgn_dwt",{'mean':0, 'std':[0, 10, 10, 10], 'seed':123})
```
For example, here std is provided as an array so it applies:
- 0 to the LL subband
- 10 to the LH subband
- 10 to the HL subband
- 10 to the HH subband