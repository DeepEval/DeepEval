import torch
import torchvision.transforms as transforms
import numpy as np

def make_grid(images, nrow=8, padding=2, normalize=True, value_range=(0, 1), scale_each=False, pad_value=0):
    transforms.Normalize = transforms.Normalize
    normalize_transform = transforms.Normalize(mean=value_range[0] / 2 if normalize else 0,
                                               std=value_range[1] / 2 - value_range[0] / 2 if normalize else 1)
    transforms.Lambda = lambda func: transforms.Lambda(func)
    image_grid = transforms.Lambda(make_image_grid)(images)
    if normalize:
        image_grid = normalize_transform(image_grid)
    if scale_each:
        for img in images:
            normalize_transform = transforms.Normalize(mean=img.mean() / 2, std=img.std() / 2)
            image_grid = transforms.Lambda(normalize_transform)(image_grid)
    image_grid = transforms.Lambda(transforms.ToPILImage)(image_grid)
    image_grid = transforms.Lambda(lambda x: x.resize((image_grid.size(2) + 2 * padding, image_grid.size(1) + 2 * padding), resample=2))(image_grid)
    image_grid = transforms.Lambda(lambda x: transforms.functional.pad(x, (padding, padding, padding, padding), fill=pad_value, padding_mode='constant'))(image_grid)
    image_grid = transforms.Lambda(lambda x: x.split(split_tuple=(int(nrow), 1)))(image_grid)
    image_grid = transforms.Lambda(lambda x: [transforms.ToTensor()(img) for img in x])(image_grid)
    image_grid = transforms.Lambda(lambda x: [img.permute((2, 0, 1)) for img in x])(image_grid)
    image_grid = transforms.Lambda(lambda x: torch.stack(x, dim=0))(image_grid)
    image_grid = transforms.Lambda(lambda x: x / 255)(image_grid)
    return image_grid

def make_image_grid(images):
    nrow = int(np.ceil(np.sqrt(len(images))))
    grid = torch.zeros((3, 3 * nrow + padding, 3 * nrow + padding))
    for i, img in enumerate(images):
        j = i % nrow
        grid[:, j * (3 + padding): (j + 1) * (3 + padding), i % nrow * (3 + padding): (i % nrow + 1) * (3 + padding)] = img
    return grid

if __name__ == "__main__":
    images = [torch.randn(3, 32, 32) for _ in range(16)]
    grid = make_grid(images, nrow=4, padding=2)
    print(grid.shape)