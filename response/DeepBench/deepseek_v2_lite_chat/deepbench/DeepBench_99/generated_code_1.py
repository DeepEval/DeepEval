import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Union, Tuple
from PIL import ImageChops, ImageEnhance, Image

def adjust_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    contrast = enhancer.enhance(0.5)
    contrast = np.array(contrast)
    return contrast

def adjust_saturation(image):
    enhancer = ImageEnhance.Color(image)
    saturation = enhancer.enhance(0.5)
    saturation = np.array(saturation)
    return saturation

def adjust_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    brightness = enhancer.enhance(0.5)
    brightness = np.array(brightness)
    return brightness

def solarization(image):
    half = image / 2
    result = ImageChops.add(image, half, threshold=256)
    result.putalpha(half)
    return result

def gaussian_blur(image, sigmax, sigmay):
    kernel_size = int(np.ceil(max(sigmax, sigmay) * 3.5))
    sigma = (sigmax, sigmay)
    kernel = torchvision.transforms.GaussianBlur2d(kernel_size, sigma, align_corners=False)
    return kernel(image)

def normalize(image):
    mean, std = imagenet_normalize
    return transforms.functional.normalize(image, mean=mean, std=std)

class DINOCollateFunction(torchvision.datasets.folder.DefaultFolder):
    def __init__(self, global_crop_size=224, global_crop_scale=(0.4, 1.0), local_crop_size=96, local_crop_scale=(0.05, 0.4), n_local_views=6, hf_prob=0.5, vf_prob=0, rr_prob=0, rr_degrees: Optional[Union[float, Tuple[float, float]]] = None, cj_prob=0.8, cj_bright=0.4, cj_contrast=0.4, cj_sat=0.2, cj_hue=0.1, random_gray_scale=0.2, gaussian_blur=(1.0, 0.1, 0.5), kernel_size: Optional[float] = None, kernel_scale: Optional[float] = None, sigmas: Tuple[float, float] = (0.1, 2), solarization_prob=0.2, normalize=imagenet_normalize, ):
        super(DINOCollateFunction, self).__init__()
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize(global_crop_size),
            transforms.ColorJitter(cj_prob, cj_bright, cj_contrast, cj_sat, cj_hue),
            transforms.RandomHorizontalFlip(hf_prob),
            transforms.RandomVerticalFlip(vf_prob),
            transforms.RandomRotation(rr_prob, range=(-rr_degrees, rr_degrees)),
            transforms.Grayscale(random_gray_scale),
            transforms.GaussianBlur(kernel_size, sigma=sigmas),
            solarization if solarization_prob > 0 else transforms.Lambda(lambda x: x),
            normalize,
            transforms.Resize(local_crop_size),
            transforms.ColorJitter(cj_prob, cj_bright, cj_contrast, cj_sat, cj_hue) if local_crop_scale != (0.05, 0.4) else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip(hf_prob) if hf_prob > 0 else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip(vf_prob) if vf_prob > 0 else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(rr_prob, range=(-rr_degrees, rr_degrees)) if rr_prob > 0 else transforms.Lambda(lambda x: x),
            transforms.Grayscale(random_gray_scale) if random_gray_scale > 0 else transforms.Lambda(lambda x: x),
            transforms.GaussianBlur(kernel_size, sigma=sigmas) if kernel_scale != (0.1, 0.5) else transforms.Lambda(lambda x: x),
            transforms.Resize(local_crop_size) if local_crop_scale != (0.4, 1.0) else transforms.Lambda(lambda x: x),
        ])

    def __call__(self, x):
        return [self.transform(img) for img in x]

if __name__ == "__main__":
    # Sample input values
    data_loader = torch.utils.data.DataLoader([
        {'image': Image.open('sample.jpg').convert('RGB'), 'mask': Image.open('mask.jpg').convert('L')}
    ], batch_size=2)
    
    collate = DINOCollateFunction()
    
    for batch in data_loader:
        inputs = collate(batch)
        print(inputs)