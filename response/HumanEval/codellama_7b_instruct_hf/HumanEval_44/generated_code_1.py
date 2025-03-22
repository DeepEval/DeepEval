from decimal import Decimal
import math

def method(x, base):
    if not (isinstance(x, int) or isinstance(x, float)):
        raise TypeError("Input number must be a number")
    if not isinstance(base, int):
        raise TypeError("Base must be an integer")
    if base < 2 or base > 10:
        raise ValueError("Base must be between 2 and 10")

    output = ""
    if x < 0:
        output += "-"
        x = -x
    if x == 0:
        output += "0"

    while x > 0:
        remainder = int(x % base)
        output = str(remainder) + output
        x = math.floor(x / base)

    return output

# Test case
x = 10
base = 2
print(method(x, base)) # Output: "1010"