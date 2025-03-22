def method(numbers):
    total_sum = 0
    for index, value in enumerate(numbers):
        if index % 3 == 0:
            total_sum += value ** 2  # Square if index is a multiple of 3
        elif index % 4 == 0:
            total_sum += value ** 3  # Cube if index is a multiple of 4 and not a multiple of 3
        else:
            total_sum += value  # Keep the value unchanged for other indices
    return total_sum

# Test case for validation
test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
output = method(test_list)
print("The output for the test case is:", output)