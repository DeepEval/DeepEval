from math import log2
def method(N):
    binary_str = bin(N)[2:]
    total_sum = 0
    for digit in binary_str:
        total_sum += int(digit)

    return total_sum

# Test case
N = 12345
expected_sum = 15
actual_sum = method(N)
print(f"Test case: N = {N}, expected_sum = {expected_sum}, actual_sum = {actual_sum}")