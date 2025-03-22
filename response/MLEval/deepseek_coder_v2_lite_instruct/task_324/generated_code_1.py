import time

def method():
    # Simulate a 50-second task
    time.sleep(50)
    
    # Define the output
    output = "Task completed successfully"
    
    return output

# Call the method for validation
result = method()
print(result)