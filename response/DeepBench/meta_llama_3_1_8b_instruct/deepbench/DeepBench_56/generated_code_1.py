import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Union, List

def encode_jpeg(images: Union[np.ndarray, List[np.ndarray]], quality: int) -> Union[np.ndarray, List[np.ndarray]]:
    if not 1 <= quality <= 100:
        raise ValueError("Quality must be between 1 and 100")
    if not images:
        raise ValueError("Input cannot be an empty list")

    if isinstance(images, np.ndarray):
        images = [images]

    outputs = []
    for image in images:
        image = (image * 255).astype(np.uint8).transpose((1, 2, 0))
        pil_image = Image.fromarray(image)
        output = transforms.ToTensor()(pil_image.resize((pil_image.size[0] // 2, pil_image.size[1] // 2)))
        output = transforms.ToTensor()(transforms.ToPILImage(mode='RGB')(output) \
                                       .convert('RGB') \
                                       .save('temp.jpg', quality=quality, subsampling=0) \
                                       .convert('RGB') \
                                       .resize((pil_image.size[0] // 2, pil_image.size[1] // 2)) \
                                       .save('temp.jpg', quality=quality, subsampling=0) \
                                       .convert('RGB'))
        output = output.numpy()[..., ::-1]
        outputs.append(output)

    return np.array(outputs)

if __name__ == "__main__":
    import numpy as np
    import torch

    # Create sample input values
    images = np.random.rand(2, 3, 256, 256)
    quality = 75

    # Call the function
    outputs = encode_jpeg(images, quality)

    # Print the results
    for i, output in enumerate(outputs):
        print(f"Output {i+1} shape: {output.shape}")
        print(f"Output {i+1} type: {output.dtype}")
        print()