import torch
from torchvision import transforms

def encode_jpeg(images, quality=95):
    if not 1 <= quality <= 100:
        raise ValueError("Quality must be between 1 and 100")
    if not isinstance(images, (list, tuple)):
        images = [images]
    
    encoded_images = []
    for image in images:
        if image.ndim != 3 or image.shape[0] not in [1, 3]:
            raise ValueError("Images must be in CHW layout with C being 1 or 3")
        if image.shape[1] < 0 or image.shape[2] < 0:
            raise ValueError("Image dimensions must be positive")
        
        transform = transforms.ToPILImage()
        pil_image = transform(image)
        byte_array = bytearray()
        pil_image.save(byte_array, format='JPEG', quality=quality)
        encoded_images.append(torch.tensor(byte_array))
    
    if len(encoded_images) == 1:
        return encoded_images[0]
    else:
        return encoded_images

if __name__ == "__main__":
    # Example usage
    image = torch.rand(3, 256, 256)  # Random CHW image
    encoded = encode_jpeg(image, quality=80)
    print(encoded.shape)  # Should be a single tensor with shape [num_bytes]

    images = [torch.rand(3, 256, 256), torch.rand(3, 256, 256)]  # List of CHW images
    encoded_list = encode_jpeg(images, quality=90)
    for encoded_image in encoded_list:
        print(encoded_image.shape)  # Each should be [num_bytes]