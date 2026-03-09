import torch
import torch.nn.functional as F

def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    N, C, H, W = input.size()
    out_channels, in_channels, kernel_height, kernel_width = weight.size()

    if mask is not None:
        mask = mask.view(1, 1, mask.size(0), mask.size(1)).expand(N, out_channels, -1, -1)

    # Compute output size
    out_height = (H + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (W + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1

    # Prepare the grid for sampling
    offset = offset.permute(0, 2, 3, 1)  # N x H x W x 2C
    grid = torch.empty(N, out_height, out_width, 2, dtype=input.dtype, device=input.device)

    for i in range(out_height):
        for j in range(out_width):
            grid[:, i, j, 0] = (j * stride) + offset[:, i, j, 0]
            grid[:, i, j, 1] = (i * stride) + offset[:, i, j, 1]

    grid = grid.permute(0, 3, 1, 2)  # N x 2 x out_height x out_width
    output = F.grid_sample(input, grid, align_corners=True)

    output = F.conv2d(output, weight, bias=bias, stride=1, padding=0, dilation=dilation)

    if mask is not None:
        output = output * mask

    return output

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 5, 5)  # N, C, H, W
    offset_tensor = torch.randn(1, 18, 5, 5)  # N, 2 * C, H, W
    weight_tensor = torch.randn(3, 3, 3, 3)  # out_channels, in_channels, kernel_height, kernel_width
    bias_tensor = torch.randn(3)  # out_channels
    mask_tensor = torch.randn(1, 1, 5, 5)  # Optional mask

    output = deform_conv2d(input_tensor, offset_tensor, weight_tensor, bias=bias_tensor, mask=mask_tensor)
    print(output)