import random
from typing import Optional, Tuple, Union
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

class DINOCollateFunction(MultiViewCollateFunction):
    def __init__(self, global_crop_size=224, global_crop_scale=(0.4, 1.0), local_crop_size=96,
  local_crop_scale=(0.05, 0.4), n_local_views=6, hf_prob=0.5, vf_prob=0, rr_prob=0,
  rr_degrees: Optional[Union[float, Tuple[float, float]]] = None, cj_prob=0.8, cj_bright=0.4,
  cj_contrast=0.4, cj_sat=0.2, cj_hue=0.1, random_gray_scale=0.2, gaussian_blur=(1.0,
  0.1, 0.5), kernel_size: Optional[float] = None, kernel_scale: Optional[float] =
  None, sigmas: Tuple[float, float] = (0.1, 2), solarization_prob=0.2, normalize=imagenet_normalize,
  ):
        super().__init__()
        self.global_crop_size = global_crop_size
        self.global_crop_scale = global_crop_scale
        self.local_crop_size = local_crop_size
        self.local_crop_scale = local_crop_scale
        self.n_local_views = n_local_views
        self.hf_prob = hf_prob
        self.vf_prob = vf_prob
        self.rr_prob = rr_prob
        self.rr_degrees = rr_degrees
        self.cj_prob = cj_prob
        self.cj_bright = cj_bright
        self.cj_contrast = cj_contrast
        self.cj_sat = cj_sat
        self.cj_hue = cj_hue
        self.random_gray_scale = random_gray_scale
        self.gaussian_blur = gaussian_blur
        self.kernel_size = kernel_size
        self.kernel_scale = kernel_scale
        self.sigmas = sigmas
        self.solarization_prob = solarization_prob
        self.normalize = normalize

        self.global_transform = T.Compose([
            T.RandomResizedCrop(size=self.global_crop_size, scale=self.global_crop_scale),
            T.RandomHorizontalFlip(p=self.hf_prob),
            T.RandomVerticalFlip(p=self.vf_prob),
            T.RandomRotation(degrees=self.rr_degrees, p=self.rr_prob),
            T.ColorJitter(brightness=self.cj_bright, contrast=self.cj_contrast,
  saturation=self.cj_sat, hue=self.cj_hue, p=self.cj_prob),
            T.RandomGrayscale(p=self.random_gray_scale),
            T.GaussianBlur(kernel_size=self.kernel_size, sigma=(self.gaussian_blur[0],
  self.gaussian_blur[1]), p=self.gaussian_blur[2]),
            T.Solarize(p=self.solarization_prob),
            self.normalize
        ])

        self.local_transform = T.Compose([
            T.RandomCrop(size=self.local_crop_size, scale=self.local_crop_scale),
            T.RandomHorizontalFlip(p=self.hf_prob),
            T.RandomVerticalFlip(p=self.vf_prob),
            T.RandomRotation(degrees=self.rr_degrees, p=self.rr_prob),
            T.ColorJitter(brightness=self.cj_bright, contrast=self.cj_contrast,
  saturation=self.cj_sat, hue=self.cj_hue, p=self.cj_prob),
            T.RandomGrayscale(p=self.random_gray_scale),
            T.GaussianBlur(kernel_size=self.kernel_size, sigma=(self.gaussian_blur[0],
  self.gaussian_blur[1]), p=self.gaussian_blur[2]),
            T.Solarize(p=self.solarization_prob),
            self.normalize
        ])

if __name__ == "__main__":
    cf = DINOCollateFunction()
    sample_input = {'image': torch.randn(1, 3, 224, 224)} 
    output = cf(sample_input)
    print(output)