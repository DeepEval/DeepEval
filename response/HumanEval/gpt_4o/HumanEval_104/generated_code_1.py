def method(x):
    def has_even_digit(n):
        # Convert number to string to check each digit
        return any(int(digit) % 2 == 0 for digit in str(n))

    # Filter out numbers that contain any even digit
    filtered_numbers = [num for num in x if not has_even_digit(num)]

    # Return the sorted list of filtered numbers
    return sorted(filtered_numbers)


# Test case
test_list = [135, 246, 357, 468, 579, 680, 791]
expected_output = [135, 357, 579, 791]
output = method(test_list)

print("Output:", output)
print("Expected:", expected_output)
print("Test Passed:", output == expected_output)