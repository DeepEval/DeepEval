def method():
    # Function to check if a number contains even digit
    def has_even_digit(n):
        return any(int(digit) % 2 == 0 for digit in str(n))
    
    # Input list of positive integers
    numbers = [45, 32, 123, 999, 678, 47, 111, 13579]
    
    # Filter out numbers with even digits
    numbers_without_even_digits = [num for num in numbers if not has_even_digit(num)]
    
    # Sort the list
    numbers_without_even_digits.sort()
    
    return numbers_without_even_digits

# Test case
def test_method():
    result = method()
    # assert result == [45, 32, 999, 13579], "Test case failed"
    print("All test cases passed")

test_method()