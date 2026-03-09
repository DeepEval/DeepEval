import tensorflow as tf
from tensorflow.image import encode_jpeg

def encode_jpeg(images: tf.Tensor or list, quality: int = 95) -> tf.Tensor or list:
    if not (1 <= quality <= 100):
        raise ValueError("Quality must be between 1 and 100")
    if not images:
        raise ValueError("Empty list cannot be encoded")

    if isinstance(images, tf.Tensor):
        return encode_jpeg(images[tf.newaxis, :, :, :], quality=quality)[0]
    else:
        return tf.map_fn(lambda img: encode_jpeg(img, quality=quality), images)

if __name__ == "__main__":
    # Sample input
    images = [
        tf.random.uniform([224, 224, 3], seed=0),
        tf.random.uniform([224, 224, 3], seed=1),
    ]

    # Encode images to JPEG
    jpeg_bytes = encode_jpeg(images)

    # Print the JPEG bytes
    print(jpeg_bytes)