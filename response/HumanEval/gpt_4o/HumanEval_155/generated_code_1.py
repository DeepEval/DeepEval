def method(number):
    even_count = 0
    odd_count = 0
    
    # Convert number to string to iterate over each digit
    for digit in str(abs(number)):  # abs to handle negative numbers
        if int(digit) % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    
    output = (even_count, odd_count)
    return output

# Test case for validation
print(method(123456))  # Expected output: (3, 3) since there are three even digits (2, 4, 6) and three odd digits (1, 3, 5)