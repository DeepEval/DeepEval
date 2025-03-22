import time

def method():
    # Start time
    start_time = time.time()

    # Busy-wait loop to consume approximately 50 seconds
    while (time.time() - start_time) < 50:
        pass  # This loop will run until 50 seconds have passed

    # The output can be a simple message indicating the task is complete
    output = "Method run complete after approximately 50 seconds."

    return output

# Call the method for validation
print(method())