import torch
import torchvision.transforms.functional as TF
from PIL import Image
import io

def encode_jpeg(images, quality=95):
    if isinstance(images, list):
        if not images:
            raise ValueError("Input list is empty.")
        images = [torch.tensor(image) if isinstance(image, list) else image for image in images]
    elif isinstance(images, torch.Tensor):
        images = [images]
    else:
        raise ValueError("Input must be a tensor or a list of tensors.")
    
    if not (1 <= quality <= 100):
        raise ValueError("Quality must be between 1 and 100.")
    
    jpeg_images = []
    for img in images:
        img = img.permute(1, 2, 0)  # Convert from CHW to HWC format
        img_pil = Image.fromarray(img.numpy().astype('uint8'))
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        jpeg_images.append(buffer.getvalue())
    
    return jpeg_images

if __name__ == "__main__":
    # Create a sample image tensor (C, H, W)
    sample_image = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    output = encode_jpeg(sample_image, quality=85)
    print(len(output[0]), "bytes of JPEG data created.")