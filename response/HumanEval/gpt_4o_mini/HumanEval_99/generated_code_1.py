def method(value: str):
    # Convert the string value to a float
    number = float(value)
    
    # Use the built-in round function to get the nearest integer
    rounded_number = round(number)
    
    # Check if the number is equidistant from two integers
    if abs(number - rounded_number) == 0.5:
        # If it is equidistant, round away from zero
        if number > 0:
            rounded_number += 1  # round up for positive numbers
        else:
            rounded_number -= 1  # round down for negative numbers
    
    return rounded_number

# Test case
output = method("2.5")
print(output)  # Output should be 3

output = method("-2.5")
print(output)  # Output should be -3

output = method("2.3")
print(output)  # Output should be 2

output = method("-2.3")
print(output)  # Output should be -2