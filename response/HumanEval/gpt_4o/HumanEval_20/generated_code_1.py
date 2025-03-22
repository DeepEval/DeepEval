def method(numbers):
    if len(numbers) < 2:
        raise ValueError("List must contain at least two numbers.")
    
    # Sort the list of numbers
    numbers.sort()
    
    # Initialize variables to track the smallest difference and the closest pair
    smallest_diff = float('inf')
    closest_pair = (numbers[0], numbers[1])
    
    # Iterate through the sorted list to find the closest pair
    for i in range(len(numbers) - 1):
        diff = numbers[i+1] - numbers[i]
        if diff < smallest_diff:
            smallest_diff = diff
            closest_pair = (numbers[i], numbers[i+1])
    
    return closest_pair

# Example test case
numbers = [4, 9, 1, 17, 20, 25]
output = method(numbers)
print("Closest pair:", output)  # Expected output: (4, 9)