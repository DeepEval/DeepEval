import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image

def decode_jpeg(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)

def encode_jpeg(img_tensors, quality=75):
    if not isinstance(img_tensors, list):
        img_tensors = [img_tensors]
        
    encoded_images = []
    for img_tensor in img_tensors:
        if img_tensor.shape.ndim != 3 or img_tensor.shape[2] != 3:
            raise ValueError("Input images must have a shape of (height, width, channels) and a channel count of 3.")
        img = tf.image.decode_jpeg(img_tensor, channels=3)
        img = tf.image.encode_jpeg(img, quality=quality)
        encoded_images.append(img)
    return encoded_images

if __name__ == "__main__":
    import io

    # Sample input values
    img1 = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    img2 = tf.constant([[[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
    img3 = tf.constant([[[20, 21, 22], [23, 24, 25], [26, 27, 28]]])

    # Call the function and print the results
    encoded_imgs = encode_jpeg([img1, img2, img3], quality=75)
    for i, img in enumerate(encoded_imgs):
        img_bytes = tf.image.decode_jpeg(img, channels=3)
        decoded_img = decode_jpeg(img_bytes.numpy())
        print(f"Decoded image from JPEG bytes:")
        print(decoded_img)
        print(f"Encoded JPEG quality: {75}")
        print("---")