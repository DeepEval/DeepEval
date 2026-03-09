import cv2
import numpy as np

def encode_jpeg(image, quality=75):
    """
    Encodes an input tensor or list of tensors representing images into JPEG format.

    Parameters:
    image (Tensor or list of Tensors): Input image(s) in CHW layout.
    quality (int): Quality parameter between 1 and 100 to control the output JPEG quality.

    Returns:
    Tensor or list of Tensors: Raw bytes of the JPEG file(s).

    Raises:
    ValueError: If the quality is not between 1 and 100 or if an empty list is passed as input.
    """
    if not isinstance(image, (list, tuple)):
        image = [image]

    if not isinstance(quality, int) or quality < 1 or quality > 100:
        raise ValueError("Quality must be an integer between 1 and 100")

    if len(image) == 0:
        raise ValueError("Input list cannot be empty")

    # Convert image(s) to uint8
    image_uint8 = [np.uint8(img) for img in image]

    # Encode image(s) to JPEG
    jpeg_bytes = [cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, quality))[1] for img in image_uint8]

    # Convert JPEG bytes to numpy array
    jpeg_array = [np.array(jpeg_bytes) for jpeg_bytes in jpeg_bytes]

    return jpeg_array

if __name__ == "__main__":
    # Create sample input values
    image = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    quality = 80

    # Call the function
    output = encode_jpeg(image, quality)

    # Print the output
    print(output)