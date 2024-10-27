# Attack File
The attack file contains different type of attack, and can be used by the wrapper `attack function` :
- input:
    - `image_path`: the path of the image
    - `attack_name`: is the name of the function
    - `args`: the argument that attack_name takes as input
- output:
    - `img`: return the attacked image
## Features
- **Global Attacks**: Affect the entire image using Gaussian blur, median blur, additive white Gaussian noise (AWGN), JPEG compression, and sharpening.
- **Localized Attacks**: Target specific regions, such as edges, by applying transformations selectively based on edge-detection methods.
- **Combination Attacks**: Chain multiple transformations in sequence for more complex effects, such as resizing + JPEG compression.
- **Wavelet Transform Attacks**: Use Discrete Wavelet Transform (DWT) to apply attacks at different frequency levels.

## Example
Here some example of usage
### Jpeg compression
The following code snippet apply jpeg compression to `watermarked_0.bmp` file with a quality factor of 10
```
attacked= attack.attack('watermarked_0.bmp',"jpeg_compression", {'qf':10})
```

### Blur edge
The following code snippet apply a blurring only on the edges of the image, in order to increase the wpsnr. The edge_func represent function used to detect the edge, 0 sobel, 1 is canny.
```
attacked= attack.attack('watermarked_0.bmp',"gauss_edge",{ 'sigma':0.4,'edge_func':0})
```
