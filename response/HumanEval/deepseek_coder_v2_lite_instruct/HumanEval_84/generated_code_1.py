def method(N):
    # Convert the number to a binary string and remove the '0b' prefix
    binary_str = bin(N)[2:]
    
    # Calculate the sum of the digits in the binary string
    sum_of_digits = sum(int(digit) for digit in binary_str)
    
    return sum_of_digits

# Test case
N = 123
output = method(N)
print(f"The sum of the digits of {N} in binary is: {output}")