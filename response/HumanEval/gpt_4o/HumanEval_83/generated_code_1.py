def method(n):
    if n == 1:
        # Special case: single-digit numbers, i.e., [1]
        return 1
    
    # Calculate numbers that start with '1'
    start_with_1 = 10**(n-1)
    
    # Calculate numbers that end with '1'
    end_with_1 = 10**(n-1)
    
    # Calculate numbers that both start and end with '1'
    both_start_and_end_with_1 = 10**(n-2)
    
    # Total count
    output = start_with_1 + end_with_1 - both_start_and_end_with_1
    return output

# Test case
print(method(1))  # Expected output: 1 (since only '1' is a 1-digit number that starts or ends with '1')
print(method(2))  # Expected output: 19 (numbers are 11, 12, ..., 19, 21, 31, ..., 91)
print(method(3))  # Additional test case