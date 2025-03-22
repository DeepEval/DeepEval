import time

def method():
    start_time = time.time()
    # Code to be timed goes here
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return output

# Call the method for validation
output = method()