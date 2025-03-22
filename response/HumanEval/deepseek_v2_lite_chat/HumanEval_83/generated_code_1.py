def method(n):
    # Total number of n-digit numbers
    total_numbers = 10 ** n  # 10^n gives the total number of n-digit numbers
    
    # Number of n-digit numbers starting with 1
    start_with_1 = 9 * 10 ** (n - 1)  # Multiply by 9 because the first digit can be any digit except 0
    
    # Number of n-digit numbers ending with 1
    end_with_1 = 10 ** (n - 1)  # Last digit can be 0, 1, 2, ..., 9
    
    # Counting numbers that start or end with 1
    count = start_with_1 + end_with_1
    
    return count

# Example usage:
# For n = 2 (two-digit numbers), the count is 18 (9 that start with 1 and 9 that end with 1)
n = 2
print(method(n))  # Should output 18