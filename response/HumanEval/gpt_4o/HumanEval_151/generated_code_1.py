def method(numbers):
    # Initialize the sum of squares
    sum_of_squares = 0
    
    # Iterate over each number in the list
    for num in numbers:
        # Check if the number is a non-negative integer and is odd
        if isinstance(num, int) and num >= 0 and num % 2 != 0:
            # Add the square of the number to the sum
            sum_of_squares += num ** 2
    
    return sum_of_squares

# Test cases to validate the solution
print(method([1, 3, 2, 0]))  # Expected output: 10 (1^2 + 3^2)
print(method([-1, -2, 0]))   # Expected output: 0
print(method([9, -2]))       # Expected output: 81 (9^2)
print(method([0]))           # Expected output: 0
print(method([]))            # Expected output: 0