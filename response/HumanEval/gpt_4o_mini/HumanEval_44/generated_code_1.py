def method(x, base):
    if base < 2 or base >= 10:
        raise ValueError("Base must be between 2 and 9 inclusive.")
    if x < 0:
        raise ValueError("Input number x must be non-negative.")

    if x == 0:
        return "0"

    result = []
    
    while x > 0:
        remainder = x % base
        result.append(str(remainder))
        x //= base

    # Reverse the result to get the correct representation
    result.reverse()
    
    # Join the list into a string
    output = ''.join(result)
    return output

# Test case
x = 255
base = 8
output = method(x, base)
print(f"The representation of {x} in base {base} is: {output}")