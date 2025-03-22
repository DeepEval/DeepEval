def double_the_difference(numbers):
    if not numbers:  # Check if the list is empty
        return 0
    
    # Initialize sum to 0
    sum_of_squares = 0
    
    # Iterate through the list
    for num in numbers:
        # Check if the number is positive and an integer
        if num > 0 and isinstance(num, int):
            # Square the number if it's odd
            if num % 2 != 0:
                sum_of_squares += num ** 2
    
    return sum_of_squares

# Test cases
print(double_the_difference([1, 3, 2, 0]))  # Output: 10
print(double_the_difference([-1, -2, 0]))   # Output: 0
print(double_the_difference([9, -2]))       # Output: 81
print(double_the_difference([0]))          # Output: 0