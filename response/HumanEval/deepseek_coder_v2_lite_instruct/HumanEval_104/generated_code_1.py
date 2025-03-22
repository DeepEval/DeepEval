def method():
    # Sample list of positive integers
    x = [345, 23, 67, 89, 1234, 56, 78, 90]
    
    # Function to check if a number has any even digit
    def has_even_digit(n):
        return any(int(digit) % 2 == 0 for digit in str(n))
    
    # Filter out numbers with even digits
    filtered_numbers = [num for num in x if not has_even_digit(num)]
    
    # Sort the filtered numbers
    sorted_numbers = sorted(filtered_numbers)
    
    # Return the sorted list
    return sorted_numbers

# Test the function
output = method()
print(output)  # Expected output: [23, 67, 89, 1234]