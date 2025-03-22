from collections import Counter

def method(nums):
    # Count the frequency of each integer in the list
    frequency = Counter(nums)
    
    # Initialize the variable to store the greatest integer meeting the condition
    greatest_integer = -1
    
    # Iterate through the frequency dictionary
    for num in frequency:
        if frequency[num] >= num:
            greatest_integer = max(greatest_integer, num)
    
    return greatest_integer

# Test case
test_case = [3, 3, 3, 2, 2, 2, 1]
output = method(test_case)
print(f"Test case result: {output}")  # Expected output: 3