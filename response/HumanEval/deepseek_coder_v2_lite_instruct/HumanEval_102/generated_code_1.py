def method():
    # Example test case
    x = 3
    y = 10
    output = find_largest_even(x, y)
    return output

def find_largest_even(x, y):
    largest_even = -1
    for num in range(x, y + 1):
        if num % 2 == 0:
            if num > largest_even:
                largest_even = num
    return largest_even

# Test case
x = 3
y = 10
expected_output = 10
actual_output = find_largest_even(x, y)
# assert actual_output == expected_output, f"Test failed! Expected {expected_output}, but got {actual_output}"

# Run the method to see the result
print(method())  # Output should be 10