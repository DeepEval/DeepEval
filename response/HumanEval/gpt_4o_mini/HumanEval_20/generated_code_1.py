def method(numbers):
    # Step 1: Sort the list of numbers
    sorted_numbers = sorted(numbers)
    
    # Step 2: Initialize variables to keep track of the closest pair
    min_diff = float('inf')
    closest_pair = (0, 0)
    
    # Step 3: Iterate through sorted list to find the closest pair
    for i in range(len(sorted_numbers) - 1):
        num1 = sorted_numbers[i]
        num2 = sorted_numbers[i + 1]
        diff = num2 - num1
        
        if diff < min_diff:
            min_diff = diff
            closest_pair = (num1, num2)
    
    return closest_pair

# Test case for validation
test_numbers = [4, 1, 3, 7, 5, 2]
output = method(test_numbers)
print(output)  # Expected output: (1, 2) or (2, 3) depending on the closest pair found