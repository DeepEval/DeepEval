import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage

def to_pil_image(pic, mode=None):
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point():
            pic = torch.clamp(pic, 0, 1)
        to_tensor = ToTensor()
        img = to_tensor(pic)
    elif isinstance(pic, np.ndarray):
        img = ToPILImage(mode=mode)(pic)
    else:
        raise ValueError(f"Unsupported input type: {type(pic)}")

    return img

if __name__ == "__main__":
    # Create sample input values
    tensor = torch.randn(3, 256, 256)
    ndarray = np.random.randint(0, 256, (256, 256, 3))
    image_path = "path_to_your_image.jpg"  # Replace with your image path

    # Call the function and print the results
    pil_image_from_tensor = to_pil_image(tensor)
    pil_image_from_ndarray = to_pil_image(ndarray, mode="RGB")
    pil_image_from_path = to_pil_image(image_path)

    # Save the images
    pil_image_from_tensor.save("tensor_image.png")
    pil_image_from_ndarray.save("ndarray_image.png")
    try:
        pil_image_from_path.save("image_from_path.png")
    except:
        print("Error: Failed to load the image from the specified path.")

    # Display the images
    pil_image_from_tensor.show()
    pil_image_from_ndarray.show()
    # pil_image_from_path.show()  # Commented out as it may raise an exception