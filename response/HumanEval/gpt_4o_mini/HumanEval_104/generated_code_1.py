def has_even_digit(num):
    """Check if the number has any even digit."""
    for digit in str(num):
        if int(digit) % 2 == 0:  # Check if digit is even
            return True
    return False

def method():
    # Sample list of positive integers
    x = [3, 7, 15, 22, 39, 41, 55, 79, 80]
    
    # Filter out numbers that have any even digit
    filtered_numbers = [num for num in x if not has_even_digit(num)]
    
    # Sort the remaining numbers
    output = sorted(filtered_numbers)
    
    return output

# Test case for validation
if __name__ == "__main__":
    print(method())  # Expected output should be a sorted list of numbers with no even digits