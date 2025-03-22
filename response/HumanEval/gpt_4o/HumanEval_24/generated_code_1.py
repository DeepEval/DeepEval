def method(n):
    # Start checking from n-1 down to 1
    for i in range(n-1, 0, -1):
        if n % i == 0:
            return i
    return None  # In case n is less than or equal to 1, there is no valid divisor.

# Test case
print(method(100))  # Expected output: 50