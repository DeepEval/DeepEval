import torch
import math

def filter3d(input, kernel, border_type, normalized):
    B, C, D, H, W = input.shape
    _, _, Kd, Kh, Kw = kernel.shape

    # Apply padding according to the specified mode
    if border_type == 'constant':
        input_padded = torch.nn.functional.pad(input, (Kd//2, Kd-Kd//2, Kh//2, Kh-Kh//2, Kw//2, Kw-Kw//2), value=0)
    elif border_type == 'replicate':
        input_padded = torch.nn.functional.pad(input, (Kd//2, Kd-Kd//2, Kh//2, Kh-Kh//2, Kw//2, Kw-Kw//2))
    elif border_type == 'reflect':
        input_padded = torch.nn.functional.pad(input, (Kd//2, Kd-Kd//2, Kh//2, Kh-Kh//2, Kw//2, Kw-Kw//2))
    else:
        raise ValueError('Invalid border_type')

    # Convolve the kernel with the padded input tensor
    output = torch.zeros((B, C, D, H, W))
    for b in range(B):
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        for kd in range(Kd):
                            for kh in range(Kh):
                                for kw in range(Kw):
                                    output[b, c, d, h, w] += input_padded[b, c, d+kd, h+kh, w+kw] * kernel[b, c, kd, kh, kw]

    # Normalize the kernel if required
    if normalized:
        output /= math.sqrt(float(Kd * Kh * Kw))

    return output

if __name__ == "__main__":
    # Create sample input values
    input = torch.tensor([[[[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]],

                         [[10, 11, 12],
                          [13, 14, 15],
                          [16, 17, 18]],

                         [[19, 20, 21],
                          [22, 23, 24],
                          [25, 26, 27]]]], dtype=torch.float32)

    # Create sample kernel values
    kernel = torch.tensor([[[[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]],

                         [[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],

                         [[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]]], dtype=torch.float32)

    # Call the function and print the results
    output = filter3d(input, kernel, 'constant', False)
    print(output)