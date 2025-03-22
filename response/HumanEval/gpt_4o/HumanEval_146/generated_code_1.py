def method(numbers):
    def is_odd_digit(digit):
        return digit in {'1', '3', '5', '7', '9'}
    
    count = 0
    for number in numbers:
        if number > 10:
            number_str = str(number)
            if is_odd_digit(number_str[0]) and is_odd_digit(number_str[-1]):
                count += 1
    
    return count

# Test case for validation
test_numbers = [15, 23, 37, 49, 56, 78, 81, 91, 109, 135, 246]
output = method(test_numbers)
print(output)  # Expected output: 5 (15, 37, 49, 81, 135)