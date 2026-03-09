import torchvision.transforms as T
from torchvision.transforms import functional as F
from typing import Optional, Union, Tuple

class DINOCollateFunction:
    def __init__(self, global_crop_size=224, global_crop_scale=(0.4, 1.0), local_crop_size=96,
  local_crop_scale=(0.05, 0.4), n_local_views=6, hf_prob=0.5, vf_prob=0, rr_prob=0,
  rr_degrees: Optional[Union[float, Tuple[float, float]]] = None, cj_prob=0.8, cj_bright=0.4,
  cj_contrast=0.4, cj_sat=0.2, cj_hue=0.1, random_gray_scale=0.2, gaussian_blur=(1.0,
  0.1, 0.5), kernel_size: Optional[float] = None, kernel_scale: Optional[float] =
  None, sigmas: Tuple[float, float] = (0.1, 2), solarization_prob=0.2, normalize=imagenet_normalize):
        self.transform = T.Compose([
            T.RandomResizedCrop(global_crop_size, scale=global_crop_scale),
            T.RandomHorizontalFlip(hf_prob),
            T.RandomVerticalFlip(vf_prob),
            T.RandomRotation(degrees=rr_degrees) if rr_prob > 0 else T.Lambda(lambda
  x: x),
            T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue) if cj_prob >
  0 else T.Lambda(lambda x: x),
            T.RandomGrayscale(random_gray_scale) if random_gray_scale > 0 else
  T.Lambda(lambda x: x),
            T.GaussianBlur(kernel_size, sigma=gaussian_blur[0]) if gaussian_blur[0]
  > 0 else T.Lambda(lambda x: x),
            T.Solarize(threshold=256 * solarization_prob) if solarization_prob
  > 0 else T.Lambda(lambda x: x),
            normalize
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_crop_size, scale=local_crop_scale),
            T.RandomHorizontalFlip(hf_prob),
            T.RandomVerticalFlip(vf_prob),
            T.RandomRotation(degrees=rr_degrees) if rr_prob > 0 else T.Lambda(lambda
  x: x),
            T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue) if cj_prob >
  0 else T.Lambda(lambda x: x),
            T.RandomGrayscale(random_gray_scale) if random_gray_scale > 0 else
  T.Lambda(lambda x: x),
            T.GaussianBlur(kernel_size, sigma=gaussian_blur[1]) if gaussian_blur[1]
  > 0 else T.Lambda(lambda x: x),
            T.Solarize(threshold=256 * solarization_prob) if solarization_prob
  > 0 else T.Lambda(lambda x: x),
            normalize
        ])

if __name__ == "__main__":
    sample_input = [torch.randn(3, 256, 256) for _ in range(3)]
    collate_fn = DINOCollateFunction()
    output = collate_fn(sample_input)
    print(output)