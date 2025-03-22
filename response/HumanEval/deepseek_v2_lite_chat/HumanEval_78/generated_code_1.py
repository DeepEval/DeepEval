def is_prime(digit):
    """Check if the given digit is a prime hexadecimal digit."""
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    return digit in digits and digit in ['2', '3', '5', '7', 'B', 'D']

def method(hex_string):
    """Count the number of prime hexadecimal digits in the given hexadecimal string."""
    return sum(is_prime(digit) for digit in hex_string if digit in ['2', '3', '5', '7', 'B', 'D'])

# Test case
# def test_method():
#     assert method('2357AB') == 5
#     assert method('1234') == 0
#     assert method('') == 0
#     assert method('ZZZ') == 0
#     assert method('2357BD1234') == 6
print(method('2357AB'))

# Call the test function
# test_method()