import torch
import torch.nn.functional as F

def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    b, c, h, w = input.shape
    out_c, _, kh, kw = weight.shape

    # Compute output dimensions
    oh = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    ow = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1

    # Apply padding
    input_padded = F.pad(input, [padding, padding, padding, padding])

    # Prepare offset and mask
    offset = offset.view(b, -1, oh, ow, 2)
    if mask is not None:
        mask = mask.view(b, 1, oh, ow, kh, kw).expand(b, c, oh, ow, kh, kw)

    # Create a grid
    grid_y, grid_x = torch.meshgrid(torch.arange(oh), torch.arange(ow))
    grid = torch.stack((grid_x, grid_y), 2).type_as(input).unsqueeze(0).unsqueeze(0)

    # Apply offset
    offset = offset + grid
    offset = offset.permute(0, 4, 2, 3, 1)

    # Deformable convolution operation
    output = torch.zeros(b, out_c, oh, ow).type_as(input)
    for i in range(kh):
        for j in range(kw):
            offset_i_j = offset[..., i, j, :, :].long()
            mask_value = mask[..., i, j, :, :] if mask is not None else 1
            for n in range(b):
                for c_out in range(out_c):
                    weight_value = weight[c_out, :, i, j]
                    sampled_input = input_padded[n, :, 
                                                 offset_i_j[n, 1] + i, 
                                                 offset_i_j[n, 0] + j]
                    output[n, c_out] += mask_value[n] * sampled_input * weight_value

    if bias is not None:
        output += bias.view(1, -1, 1, 1)
        
    return output

if __name__ == "__main__":
    input_tensor = torch.rand(1, 1, 5, 5)
    offset_tensor = torch.rand(1, 18, 3, 3)  # Assuming 3x3 kernel
    weight_tensor = torch.rand(1, 1, 3, 3)
    bias_tensor = torch.rand(1)

    output = deform_conv2d(input_tensor, offset_tensor, weight_tensor, bias_tensor, stride=1, padding=1)
    print(output)