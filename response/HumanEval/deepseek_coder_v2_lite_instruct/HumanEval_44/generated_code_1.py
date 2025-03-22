def method():
    # Example input
    x = 255  # The number to be converted
    base = 16  # The base to which the number will be converted
    
    # Convert the number to the specified base
    if base < 2 or base > 9:
        return "Base must be between 2 and 9."
    
    # Use Python's built-in function to convert the number to the specified base
    output = ""
    while x > 0:
        remainder = x % base
        output = str(remainder) + output
        x = x // base
    
    return output

# Test case
print(method())  # Expected output: "11111111" when converting 255 to base 2