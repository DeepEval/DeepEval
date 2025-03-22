from fractions import Fraction

def method(x, n):
    # Convert the string representations of the fractions into Fraction objects
    fraction_x = Fraction(x)
    fraction_n = Fraction(n)
    
    # Multiply the two fractions
    result_fraction = fraction_x * fraction_n
    
    # Check if the result is a whole number
    # A fraction is a whole number if its denominator is 1
    is_whole_number = result_fraction.denominator == 1
    
    return is_whole_number

# Test cases
print(method("1/5", "5/1"))  # Expected output: True
print(method("1/6", "2/1"))  # Expected output: False
print(method("7/10", "10/2"))  # Expected output: False