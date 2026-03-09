import torch
import torchvision.transforms.functional as F
from PIL import Image
import io

def encode_jpeg(images, quality):
    if not (1 <= quality <= 100):
        raise ValueError("Quality must be between 1 and 100")
    
    if isinstance(images, list):
        if not images:
            raise ValueError("Input list cannot be empty")
        return [encode_single_image(img, quality) for img in images]
    else:
        return encode_single_image(images, quality)
    
def encode_single_image(image, quality):
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    # Convert CHW to HWC
    image = image.permute(1, 2, 0)
    
    # Convert to PIL Image
    image_pil = F.to_pil_image(image)
    
    # Save to JPEG
    buffer = io.BytesIO()
    image_pil.save(buffer, format='JPEG', quality=quality)
    jpeg_bytes = torch.tensor(list(buffer.getvalue()), dtype=torch.uint8)
    
    return jpeg_bytes

if __name__ == "__main__":
    # Create a dummy image tensor (CHW format)
    dummy_image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    
    # Encode single image
    jpeg_bytes = encode_jpeg(dummy_image, quality=75)
    print(f"Encoded JPEG bytes (single image): {jpeg_bytes[:10]}...")  # Print first 10 bytes for brevity
    
    # Encode a list of images
    dummy_images = [torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8) for _ in range(5)]
    jpeg_bytes_list = encode_jpeg(dummy_images, quality=80)
    print(f"Encoded JPEG bytes (first image in list): {jpeg_bytes_list[0][:10]}...")  # Print first 10 bytes for brevity