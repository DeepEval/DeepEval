import numpy as np

def do_conv2d(conv, input_, padding=None, weight=None, bias=None):
    if padding is None:
        padding = conv['padding']
    if weight is None:
        weight = conv['weight']
    if bias is None:
        bias = conv['bias']

    input_pad = np.pad(input_, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    output_shape = (input_.shape[0], weight.shape[0], (input_.shape[2] + 2 * padding - weight.shape[2]) // conv['stride'] + 1, (input_.shape[3] + 2 * padding - weight.shape[3]) // conv['stride'] + 1)
    output = np.zeros(output_shape)

    for i in range(output_shape[2]):
        for j in range(output_shape[3]):
            for k in range(output_shape[1]):
                for l in range(output_shape[0]):
                    output[l, k, i, j] = np.sum(input_pad[l, :, i*conv['stride']:i*conv['stride']+weight.shape[2], j*conv['stride']:j*conv['stride']+weight.shape[3]] * weight[k]) + bias[k]

    return output

if __name__ == "__main__":
    # Sample input values
    input_ = np.array([[[[1, 2, 3, 4, 5],
                         [5, 6, 7, 8, 9],
                         [9, 8, 7, 6, 5],
                         [5, 4, 3, 2, 1],
                         [1, 2, 3, 4, 5]]]])
    weight = np.array([[[[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]]]])
    bias = np.array([1])
    conv = {'padding': 1, 'stride': 2}

    # Call the function and print the results
    result = do_conv2d(conv, input_, weight=weight, bias=bias)
    print(result)