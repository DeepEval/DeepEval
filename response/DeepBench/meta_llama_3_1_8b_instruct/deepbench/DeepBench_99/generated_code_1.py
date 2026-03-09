import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from typing import Optional, Union, Tuple
from transformers import AutoFeatureExtractor

def function_name(args):
    global_crop_size = args.global_crop_size
    global_crop_scale = args.global_crop_scale
    local_crop_size = args.local_crop_size
    local_crop_scale = args.local_crop_scale
    n_local_views = args.n_local_views
    hf_prob = args.hf_prob
    vf_prob = args.vf_prob
    rr_prob = args.rr_prob
    rr_degrees = args.rr_degrees
    cj_prob = args.cj_prob
    cj_bright = args.cj_bright
    cj_contrast = args.cj_contrast
    cj_sat = args.cj_sat
    cj_hue = args.cj_hue
    random_gray_scale = args.random_gray_scale
    gaussian_blur = args.gaussian_blur
    kernel_size = args.kernel_size
    kernel_scale = args.kernel_scale
    sigmas = args.sigmas
    solarization_prob = args.solarization_prob
    normalize = args.normalize

    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(global_crop_size, scale=global_crop_scale, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=hf_prob),
        transforms.RandomVerticalFlip(p=vf_prob),
        transforms.RandomRotation(rr_prob, degrees=rr_degrees, expansion=False),
        transforms.ColorJitter(brightness=cj_bright, contrast=cj_contrast, saturation=cj_sat, hue=cj_hue),
        transforms.RandomGrayscale(p=random_gray_scale),
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=(sigmas[0], sigmas[1])),
        transforms.RandomSolarize(p=solarization_prob),
        transforms.ToTensor(),
        normalize
    ])

    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(local_crop_size, scale=local_crop_scale, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=hf_prob),
        transforms.RandomVerticalFlip(p=vf_prob),
        transforms.RandomRotation(rr_prob, degrees=rr_degrees, expansion=False),
        transforms.ColorJitter(brightness=cj_bright, contrast=cj_contrast, saturation=cj_sat, hue=cj_hue),
        transforms.RandomGrayscale(p=random_gray_scale),
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=(sigmas[0], sigmas[1])),
        transforms.RandomSolarize(p=solarization_prob),
        transforms.ToTensor(),
        normalize
    ])

    class DINOCollateFunction:
        def __init__(self, global_transform, local_transform):
            self.global_transform = global_transform
            self.local_transform = local_transform

    return DINOCollateFunction(global_transform, local_transform)

if __name__ == "__main__":
    # Create sample input values
    args = {
        'global_crop_size': 224,
        'global_crop_scale': (0.4, 1.0),
        'local_crop_size': 96,
        'local_crop_scale': (0.05, 0.4),
        'n_local_views': 6,
        'hf_prob': 0.5,
        'vf_prob': 0,
        'rr_prob': 0,
        'rr_degrees': 30,
        'cj_prob': 0.8,
        'cj_bright': 0.4,
        'cj_contrast': 0.4,
        'cj_sat': 0.2,
        'cj_hue': 0.1,
        'random_gray_scale': 0.2,
        'gaussian_blur': (1.0, 0.1, 0.5),
        'kernel_size': 5,
        'kernel_scale': 0.1,
       'sigmas': (0.1, 2),
       'solarization_prob': 0.2,
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    }

    # Call the function and print the results
    dinocollate_function = function_name(args)
    print(dinocollate_function.global_transform)
    print(dinocollate_function.local_transform)