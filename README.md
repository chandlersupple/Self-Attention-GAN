An implementation of the self-attention GAN (SAGAN) as described in "Self-Attention Generative Adversarial Networks" by Zhang et al. -- The model also utilizes a gradient penalty from the DRAGAN which improved the model's stability.

The following SAGAN trains on images from the celebA dataset. Benchmark images will be added to the 'images' list during the training process. To view an image, run 'image.array_to_img(image_array)'.
