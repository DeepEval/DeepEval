import math

def method(value):
    # Remove the sign from the value
    value = abs(int(value))
    
    # Round the value to the nearest integer
    rounded_value = math.floor(value + 0.5)
    
    # Check if the rounded value is equidistant from two integers
    if value - rounded_value == 0.5:
        # Round away from zero if equidistant
        return rounded_value if value >= 0 else -1 * rounded_value
    else:
        return rounded_value

# Test case
print(method("3.5"))  # Expected output: 4
print(method("-3.5")) # Expected output: -3
print(method("3"))    # Expected output: 3
print(method("0"))    # Expected output: 0
print(method("3.4"))  # Expected output: 3