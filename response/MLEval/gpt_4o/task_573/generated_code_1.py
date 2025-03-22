import numpy as np

def method():
    # Create an array or receive input (for the purpose of this example, let's create a random array)
    array = np.random.rand(4)  # Create a 1D array with 4 elements

    # Reshape the array to the expected shape (2, 2)
    output = array.reshape((2, 2))

    return output

# Call the method for validation
output = method()
print("Output:\n", output)
print("Shape of the output:", output.shape)