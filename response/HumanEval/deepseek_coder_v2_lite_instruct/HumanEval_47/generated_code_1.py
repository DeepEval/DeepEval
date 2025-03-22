import statistics

def method():
    # Example list
    l = [3, 1, 2, 4, 5]
    
    # Calculate the median
    median = statistics.median(l)
    
    # Return the median
    return median

# Test case
output = method()
print("Median:", output)  # Expected output: Median: 3