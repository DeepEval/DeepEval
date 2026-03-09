from PIL import Image
import numpy as np

def to_pil_image(pic, mode=None):
    if isinstance(pic, np.ndarray):
        # Convert numpy array to PIL Image
        try:
            import torch
            if torch.is_tensor(pic):
                import torchvision.transforms.functional as F
                pic = F.to_PIL_image(pic)
            else:
                pic = Image.fromarray(pic)
        except Exception as e:
            raise ValueError("Failed to convert the input to PIL Image.") from e
    elif isinstance(pic, torch.Tensor):
        # Convert torch tensor to PIL Image
        try:
            pic = pic.cpu().clone().numpy()
            pic = pic.squeeze(0) if pic.ndim > 2 else pic
            pic = Image.fromarray(np.uint8(pic))
        except Exception as e:
            raise ValueError("Failed to convert the input to PIL Image.") from e
    else:
        raise TypeError("Input must be a numpy.ndarray or torch.Tensor.")

    if mode:
        pic.mode = mode

    return pic

if __name__ == "__main__":
    # Test the function
    import torch
    import numpy as np
    
    tensor = torch.randn(3, 224, 224)  # Example tensor of shape (3, 224, 224)
    array = np.random.rand(3, 224, 224)  # Example numpy array of shape (3, 224, 224)
    
    image = to_pil_image(tensor)
    print("Tensor image:", image)
    
    image = to_pil_image(array)
    print("Array image:", image)
    
    image = to_pil_image(tensor, "RGB")
    print("RGB tensor image:", image)