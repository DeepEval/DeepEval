def method(n):
    if n < 1:
        return 0
    
    # Count numbers starting with '1'
    if n == 1:
        start_with_1_count = 1  # Only the number '1'
    else:
        start_with_1_count = 10**(n-1)  # 10^(n-1) to 2*10^(n-1)-1

    # Count numbers ending with '1'
    end_with_1_count = 9 * (10**(n-2))  # 1 to 9 for the first digit and 0-9 for (n-2) digits

    # Count numbers starting and ending with '1'
    if n == 1:
        start_and_end_with_1_count = 1  # Only the number '1'
    else:
        start_and_end_with_1_count = 10**(n-2)  # 1 is fixed at start and end, (n-2) middle digits can be anything

    # Total count
    total_count = start_with_1_count + end_with_1_count - start_and_end_with_1_count

    return total_count

# Test case
output = method(3)  # For 3-digit numbers
print(output)  # Expected output: 180 (100-199 start with 1, 101,111,121,131,...,191 end with 1)