# Multimedia Data Security 2024 - University of Trento
> This is the repository for the code used during the Capture the Mark Challenge 2024 of the MDS course held by the Professor G. Boato <br/>
> Mark: 30/30

## Authors
Matteo Bordignon @Bordi00
Greta Franchi @GretaFranchi
Alessandro Perez @AlessandroPerez
Marco Wang @marco3724

## Embedding Algorithm
The embedding algorithm perform a multilevel DWT (max 3 level) on the original image, the watermark is embedded in all the sub-bands in all the levels (3 watermarks per level).
To optimize the invisibility while maintaining an high degree of robustness we make use of a modular alpha which is computed base on the level and the sub-band and, a perceptual mask of each sub-band.
We decided to use a multiplicative approach because it is the most convinient.
Steps of the embedding algorithm:
1. Perform the first level DWT on the original image
2. For each sub-band but the LL:
   1. compute the perceptual mask
   2. compute the modular alpha
   3. get best locations
   4. embed watermark
3. Compute the DWT for LL sub-band and repeat the step 2
4. repeat step 3 until level=max_level
5. compute the iDWT
6. return the watermarked image 

## Detection Algorithm
The detection algorithm follows the reverse process of the embedding.
Firstly we use the original image and the watermarked image to retrieve the watermark, the we perform the same operation with the attacked image.
On each level an avarage between the watermark is performed (3 watermarks per level) in order to limit quantization errors.
Then a similarity score between each extracted watermark and the original watermark is computed. We take in account only the extrated watermark that as the higher similarity w.r.t. the original one, if the similarity is greater or equal to 0.7, then the watermark has been found.

## Defence Strategy
We chose to use a very high alpha and go all-in for the robustness. The strategy has proved successful, we were awarded as the most robust team. 

## Attack Strategy
We used a manual approach since we had some problem during the challenge. We used a mixture of global and localized attacks and we were able to successfully attack almost all the teams with a mean wpsnr of ~42

## Reference
M. Barni, F. Bartolini and A. Piva, "Improved wavelet-based watermarking through pixel-wise masking," in IEEE Transactions on Image Processing, vol. 10, no. 5, pp. 783-791, May 2001, doi: 10.1109/83.918570.
Abstract: A watermarking algorithm operating in the wavelet domain is presented. Performance improvement with respect to existing algorithms is obtained by means of a new approach to mask the watermark according to the characteristics of the human visual system (HVS). In contrast to conventional methods operating in the wavelet domain, masking is accomplished pixel by pixel by taking into account the texture and the luminance content of all the image subbands. The watermark consists of a pseudorandom sequence which is adaptively added to the largest detail bands. As usual, the watermark is detected by computing the correlation between the watermarked coefficients and the watermarking code, and the detection threshold is chosen in such a way that the knowledge of the watermark energy used in the embedding phase is not needed, thus permitting one to adapt it to the image at hand. Experimental results and comparisons with other techniques operating in the wavelet domain prove the effectiveness of the new algorithm.
keywords: {Watermarking;Discrete wavelet transforms;Wavelet domain;Image coding;Robustness;Humans;Visual system;Pixel;Phase detection;Protection},
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=918570&isnumber=19866


