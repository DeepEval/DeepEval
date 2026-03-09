import numpy as np

def do_conv2d(conv, input_, padding=0, weight=None, bias=None):
    if weight is None:
        weight = conv['weight']
    if bias is None:
        bias = conv['bias']
    if padding is None:
        padding = conv['padding']
        
    f = weight.shape[0] // 2
    input_padded = np.pad(input_, ((f, f), (f, f)), mode='constant', constant_values=padding)
    
    output = np.zeros_like(input_)
    for i in range(f, input_padded.shape[0] - f):
        for j in range(f, input_padded.shape[1] - f):
            output[i-f:i+f+1, j-f:j+f+1] += np.sum(input_padded[i-f:i+f+1, j-f:j+f+1] * weight, axis=(0, 1))
            
    if bias is not None:
        output += bias
    return output

if __name__ == "__main__":
    # Create a sample convolutional layer
    conv = {'weight': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'bias': np.array([1, 1, 1])}
    
    # Create a sample input tensor
    input_ = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Perform the convolution operation
    result = do_conv2d(conv, input_)
    
    # Print the results
    print("Result of the convolution operation:")
    print(result)