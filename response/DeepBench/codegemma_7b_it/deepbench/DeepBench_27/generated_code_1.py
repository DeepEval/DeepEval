from PIL import Image
import numpy as np
import torch

def to_pil_image(pic, mode=None):
    """
    Converts a tensor or numpy array to a PIL Image, ensuring compatibility with expected modes and handling potential errors.

    Args:
        pic (Tensor or numpy.ndarray): The image data to be converted.
        mode (optional, str): The desired color mode for the output PIL Image. Supported modes are based on the number of channels in the image and include:
            - For 1-channel images: "L" for grayscale, "I" for 32-bit signed integer pixels, "I;16" for 16-bit unsigned integer pixels, and "F" for floating-point pixels.
            - For 2-channel images: "LA" for grayscale with alpha channel.
            - For 3-channel images: "RGB" for standard RGB, "YCbCr" for color video format, "HSV" for hue, saturation, value.
            - For 4-channel images: "RGBA" for RGB with alpha channel, "CMYK" for prepress, "RGBX" for RGB with padding.

    Returns:
        Returns a PIL Image converted from the input data.

    Error Handling:
        - Raises TypeError if the input is not a Tensor or numpy array.
        - Raises ValueError if the input image is not 2D or 3D.
        - Raises ValueError if the image has more than 4 channels.
        - Raises ValueError if the specified mode is incompatible with the input data.
        - Raises TypeError if the mode is not supported for the given input type.
    """

    if not isinstance(pic, (Tensor, np.ndarray)):
        raise TypeError("Input must be a Tensor or a numpy array.")

    if isinstance(pic, Tensor):
        # Convert tensor to numpy array
        pic = pic.numpy()

    # Ensure image is 2D or 3D
    if len(pic.shape) not in [2, 3]:
        raise ValueError("Input image must be 2D or 3D.")

    # Ensure image has at most 4 channels
    if len(pic.shape) == 3 and pic.shape[2] > 4:
        raise ValueError("Input image cannot have more than 4 channels.")

    # Determine output mode
    if mode is None:
        if len(pic.shape) == 2 or (len(pic.shape) == 3 and pic.shape[2] == 1):
            mode = "L"
        elif len(pic.shape) == 3 and pic.shape[2] == 2:
            mode = "LA"
        elif len(pic.shape) == 3 and pic.shape[2] == 3:
            mode = "RGB"
        elif len(pic.shape) == 3 and pic.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError("Unsupported image mode.")
    else:
        # Check if mode is supported
        if mode not in ["L", "LA", "RGB", "RGBA", "YCbCr", "HSV", "CMYK", "RGBX", "I", "I;16", "F"]:
            raise TypeError("Invalid mode.")

    # Create PIL Image
    if mode == "L":
        image = Image.fromarray(pic.squeeze(), mode="L")
    elif mode == "LA":
        image = Image.fromarray(pic.squeeze(), mode="LA")
    elif mode == "RGB":
        image = Image.fromarray(pic.astype(np.uint8), mode="RGB")
    elif mode == "RGBA":
        image = Image.fromarray(pic.astype(np.uint8), mode="RGBA")
    elif mode == "YCbCr":
        image = Image.fromarray(pic.astype(np.uint8), mode="YCbCr")
    elif mode == "HSV":
        image = Image.fromarray(pic.astype(np.uint8), mode="HSV")
    elif mode == "CMYK":
        image = Image.fromarray(pic.astype(np.uint8), mode="CMYK")
    elif mode == "RGBX":
        image = Image.fromarray(pic.astype(np.uint8), mode="RGBX")
    elif mode == "I":
        image = Image.fromarray(pic.squeeze(), mode="I")
    elif mode == "I;16":
        image = Image.fromarray(pic.squeeze(), mode="I;16")
    elif mode == "F":
        image = Image.fromarray(pic.squeeze(), mode="F")

    return image

if __name__ == "__main__":
    # Sample input values
    tensor = torch.randn(2, 3, 224, 224)
    numpy_array = np.random.randint(0, 256, size=(224, 224, 3))

    # Call the function
    pil_image_tensor = to_pil_image(tensor)
    pil_image_numpy = to_pil_image(numpy_array)

    # Print results
    print("PIL Image from Tensor:", pil_image_tensor)
    print("PIL Image from NumPy Array:", pil_image_numpy)