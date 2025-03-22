def method(n):
    # Iterate from n-1 down to 1
    for i in range(n - 1, 0, -1):
        if n % i == 0:  # Check if i divides n evenly
            return i  # Return the first (largest) number that divides n evenly

# Test case for validation
n = 10  # Example input
output = method(n)
print(f"The largest number that divides {n} evenly and is smaller than {n} is: {output}")