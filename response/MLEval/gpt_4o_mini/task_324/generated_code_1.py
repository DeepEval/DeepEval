import numpy as np
import time

def method():
    # Start timing
    start_time = time.time()

    # Simulate a time-consuming operation
    # For example, we can create a large array and perform some computations
    large_array = np.random.rand(1000000)  # Generating a large array of random numbers
    result = np.sum(np.sqrt(large_array))  # Performing a computational task

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print elapsed time to ensure it takes about 50 seconds
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Assuming 'output' is meant to be the result of our computation
    output = result
    return output

# Call the method for validation
output = method()
print("Output:", output)