import time

def method():
    start_time = time.time()

    # Replace this with your actual code that takes 50 seconds to run
    for i in range(10000000):
        # Simulate some computationally expensive operation
        result = i * i 

    end_time = time.time()
    print(f"Code execution time: {end_time - start_time} seconds")

    # Example output
    output = "Code execution completed successfully!"
    return output

# Call the method and print the output
result = method()
print(result)