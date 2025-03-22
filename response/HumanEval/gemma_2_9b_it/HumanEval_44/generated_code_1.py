def method(x, base):
    """
    Changes the numerical base of input number x to base. 
    Returns a string representation after the conversion.
    Base numbers are less than 10.
    """
    if base < 2 or base > 10:
      raise ValueError("Invalid base. Base must be between 2 and 10.")
    
    if x == 0:
      return "0"
    
    digits = "0123456789"
    result = ""

    while x > 0:
        remainder = x % base
        result = digits[remainder] + result
        x //= base

    return result 

# Test Case
input_number = 25
target_base = 3

output = method(input_number, target_base)
print(f"Conversion of {input_number} to base {target_base}: {output}")