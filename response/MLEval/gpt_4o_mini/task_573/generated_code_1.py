import numpy as np

def method():
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # Generate random data with 100 samples and 10 features
    output = np.random.rand(100, 10)
    
    return output

# Call the method and validate the output shape
output = method()
print("Output shape:", output.shape)  # Should print: Output shape: (100, 10)