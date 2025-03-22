def method(arr):
    # Check if the input array is not empty
    if not arr:
        return []

    # Calculate the sum of the first and last index values
    first_value = arr[0]
    last_value = arr[-1]
    total_sum = first_value + last_value

    # Sort the array based on the condition
    if total_sum % 2 == 0:  # Even sum
        sorted_arr = sorted(arr, reverse=True)  # Descending order
    else:  # Odd sum
        sorted_arr = sorted(arr)  # Ascending order

    return sorted_arr

# Test case
input_array = [5, 1, 8, 3, 2]  # first_value=5, last_value=2, sum=7 (odd)
output = method(input_array)
print("Sorted Output (ascending):", output)  # Expected Output: [1, 2, 3, 5, 8]

input_array_even = [4, 2, 7, 1]  # first_value=4, last_value=1, sum=5 (odd)
output_even = method(input_array_even)
print("Sorted Output (ascending):", output_even)  # Expected Output: [1, 2, 4, 7]

input_array_even2 = [6, 3, 9, 0]  # first_value=6, last_value=0, sum=6 (even)
output_even2 = method(input_array_even2)
print("Sorted Output (descending):", output_even2)  # Expected Output: [9, 6, 3, 0]