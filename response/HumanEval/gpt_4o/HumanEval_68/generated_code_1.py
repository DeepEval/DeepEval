def method(arr):
    # Initialize variables to track the smallest even value and its index
    smallest_even_value = float('inf')
    smallest_even_index = -1

    # Iterate over the array to find the smallest even value and its index
    for index, value in enumerate(arr):
        if value % 2 == 0:  # Check if the value is even
            if value < smallest_even_value:
                smallest_even_value = value
                smallest_even_index = index

    # If a smallest even value was found, return it and its index
    if smallest_even_index != -1:
        return [smallest_even_value, smallest_even_index]
    else:
        return []

# Test case
test_array = [5, 3, 8, 2, 10, 6]
result = method(test_array)
print(result)  # Expected output: [2, 3] because 2 is the smallest even value at index 3