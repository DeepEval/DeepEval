def method(hex_string):
    # Define a set of hexadecimal digits that are primes
    prime_hex_digits = {'2', '3', '5', '7', 'B', 'D'}
    
    # Initialize a counter for prime digits
    prime_count = 0
    
    # Iterate over each character in the input string
    for char in hex_string:
        # Check if the character is a prime digit
        if char in prime_hex_digits:
            prime_count += 1
    
    # Return the count of prime digits
    return prime_count

# Test case
print(method("23A5B7D"))  # Expected output is 6 since 2, 3, 5, B, 7, and D are primes