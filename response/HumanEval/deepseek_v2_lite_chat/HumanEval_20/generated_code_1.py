def method(numbers):
    # Ensure the list has at least two elements
    if len(numbers) < 2:
        return "Error: List must have at least two numbers."
    
    # Initialize variables to hold the smallest difference and the pair of numbers
    smallest_diff = float('inf')
    closest_pair = (None, None)
    
    # Iterate through all possible pairs of numbers
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            # Calculate the absolute difference between the current pair of numbers
            diff = abs(numbers[i] - numbers[j])
            
            # If the difference is smaller than the smallest difference found so far
            if diff < smallest_diff:
                smallest_diff = diff
                closest_pair = (numbers[i], numbers[j])
    
    # Return the closest pair of numbers in the original order
    return closest_pair

# Test case
def test_method():
    numbers = [5, 9, 12, 15, 20, 22, 30]
    result = method(numbers)
    print("Test Case:", numbers)
    print("Closest pair of numbers:", result)

# Call the test function
test_method()