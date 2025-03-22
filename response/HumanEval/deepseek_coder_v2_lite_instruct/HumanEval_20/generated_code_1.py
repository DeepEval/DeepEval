def method():
    # Import necessary packages (if any)
    # For this task, no additional packages are needed
    
    # Example list of numbers
    numbers = [10, 20, 30, 40, 50]
    
    # Initialize variables to store the closest pair and the smallest difference
    closest_pair = (None, None)
    smallest_diff = float('inf')
    
    # Sort the list to make it easier to find the closest pair
    numbers.sort()
    
    # Iterate through the sorted list to find the closest pair
    for i in range(len(numbers) - 1):
        diff = numbers[i + 1] - numbers[i]
        if diff < smallest_diff:
            smallest_diff = diff
            closest_pair = (numbers[i], numbers[i + 1])
    
    # Return the closest pair in the required order
    output = closest_pair
    return output

# Test case
if __name__ == "__main__":
    result = method()
    print(result)  # Expected output: (10, 20)