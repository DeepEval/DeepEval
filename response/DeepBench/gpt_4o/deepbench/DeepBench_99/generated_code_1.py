import torch
from torchvision import transforms
from typing import Optional, Union, Tuple

class MultiViewCollateFunction:
    def __init__(self):
        pass

class DINOCollateFunction(MultiViewCollateFunction):
    def __init__(
        self,
        global_crop_size=224,
        global_crop_scale=(0.4, 1.0),
        local_crop_size=96,
        local_crop_scale=(0.05, 0.4),
        n_local_views=6,
        hf_prob=0.5,
        vf_prob=0,
        rr_prob=0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        cj_prob=0.8,
        cj_bright=0.4,
        cj_contrast=0.4,
        cj_sat=0.2,
        cj_hue=0.1,
        random_gray_scale=0.2,
        gaussian_blur=(1.0, 0.1, 0.5),
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob=0.2,
        normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ):
        super().__init__()

        # Define global view transformation
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crop_scale),
            transforms.RandomHorizontalFlip(p=hf_prob),
            transforms.RandomVerticalFlip(p=vf_prob),
            transforms.RandomRotation(degrees=rr_degrees, p=rr_prob) if rr_degrees else transforms.RandomRotation(degrees=(0, 0)),
            transforms.ColorJitter(brightness=cj_bright, contrast=cj_contrast, saturation=cj_sat, hue=cj_hue),
            transforms.RandomGrayscale(p=random_gray_scale),
            transforms.GaussianBlur(kernel_size=(kernel_size or 23), sigma=sigmas) if gaussian_blur[0] else transforms.GaussianBlur(kernel_size=(kernel_size or 23), sigma=(0, 0)),
            transforms.RandomApply([transforms.Solarize(128)], p=solarization_prob),
            transforms.ToTensor(),
            normalize
        ])

        # Define local view transformation
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crop_scale),
            transforms.RandomHorizontalFlip(p=hf_prob),
            transforms.RandomVerticalFlip(p=vf_prob),
            transforms.RandomRotation(degrees=rr_degrees, p=rr_prob) if rr_degrees else transforms.RandomRotation(degrees=(0, 0)),
            transforms.ColorJitter(brightness=cj_bright, contrast=cj_contrast, saturation=cj_sat, hue=cj_hue),
            transforms.RandomGrayscale(p=random_gray_scale),
            transforms.GaussianBlur(kernel_size=(kernel_size or 23), sigma=sigmas) if gaussian_blur[1] else transforms.GaussianBlur(kernel_size=(kernel_size or 23), sigma=(0, 0)),
            transforms.ToTensor(),
            normalize
        ])

        self.n_local_views = n_local_views

    def __call__(self, image):
        global_views = [self.global_transform(image) for _ in range(2)]
        local_views = [self.local_transform(image) for _ in range(self.n_local_views)]
        return global_views + local_views


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Create a dummy image for testing
    dummy_image = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))

    # Instantiate the DINOCollateFunction
    dino_collate = DINOCollateFunction()

    # Apply the function on the dummy image
    augmented_images = dino_collate(dummy_image)

    # Print the shapes of the results
    for i, img in enumerate(augmented_images):
        print(f"View {i + 1}: {img.shape}")