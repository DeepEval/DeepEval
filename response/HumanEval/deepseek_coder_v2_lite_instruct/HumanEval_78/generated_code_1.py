def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def method():
    hex_digits = "0123456789ABCDEF"
    prime_hex_digits = [digit for digit in hex_digits if is_prime(int(digit, 16))]
    
    # input_hex = input("Enter a hexadecimal number: ")
    input_hex = "A1B2C3D4E5F6"
    count = sum(1 for char in input_hex if char in prime_hex_digits)
    
    return count

# Test case
output = method()
print(f"Number of prime hexadecimal digits: {output}")