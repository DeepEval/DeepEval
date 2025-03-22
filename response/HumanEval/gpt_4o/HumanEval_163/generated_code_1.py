def method(a, b):
    even_digits = set()
    
    for num in range(a, b + 1):
        for char in str(num):
            digit = int(char)
            if digit % 2 == 0:
                even_digits.add(digit)
    
    output = sorted(even_digits)
    return output

# Test case
a = 123
b = 145
print(method(a, b))  # Expected output: [2, 4]