from PIL import Image
import numpy as np

def to_pil_image(pic, mode=None):
    """Converts a tensor or numpy.ndarray to a PIL Image.

    Args:
        pic (Tensor or numpy.ndarray): Input image.
        mode (str, optional): Mode for PIL Image.

    Returns:
        PIL Image: PIL Image object.

    Raises:
        TypeError: If input is not a tensor or numpy.ndarray.
        ValueError: If mode is not None and not a valid PIL mode.
    """

    if not isinstance(pic, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a tensor or numpy.ndarray.")

    if isinstance(pic, torch.Tensor):
        pic = pic.numpy()

    if mode is not None and mode not in Image.MODES:
        raise ValueError("Invalid mode.")

    if pic.ndim == 2:
        pic = pic[:, :, np.newaxis]

    if pic.shape[2] == 1:
        pic = np.repeat(pic, 3, axis=2)

    if pic.shape[2] == 4:
        pic = pic[:, :, :3]

    return Image.fromarray(pic, mode)


if __name__ == "__main__":
    # Sample input values
    tensor = torch.randn(224, 224, 3)
    ndarray = np.random.randint(0, 256, size=(224, 224, 3))

    # Call the function
    pil_image_tensor = to_pil_image(tensor)
    pil_image_ndarray = to_pil_image(ndarray)

    # Print the results
    print("PIL Image from tensor:", pil_image_tensor)
    print("PIL Image from ndarray:", pil_image_ndarray)