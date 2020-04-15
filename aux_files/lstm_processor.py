"""
Process an image that we can pass to our networks.
"""
import sys
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

def process_image_path(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.)
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)
    return x

def process_image_arr(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.)
    h, w, _ = target_shape
    if image.shape != target_shape:
        image = image.reshape(target_shape)

    # Normalize and return.
    x = (image / 255.).astype(np.float32)
    return x