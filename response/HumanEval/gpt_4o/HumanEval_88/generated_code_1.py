import copy

def method(arr):
    # Create a copy of the array so the original array remains unchanged
    copied_arr = copy.deepcopy(arr)
    
    # Calculate the sum of the first and last elements
    if len(arr) > 0:
        sum_first_last = arr[0] + arr[-1]
    else:
        return copied_arr  # Return an empty array if the input is empty

    # Sort the copied array based on the sum
    if sum_first_last % 2 == 0:
        # Even sum, sort in descending order
        copied_arr.sort(reverse=True)
    else:
        # Odd sum, sort in ascending order
        copied_arr.sort()

    return copied_arr

# Test case
original_array = [3, 1, 2, 4, 5]
sorted_array = method(original_array)

print("Original array:", original_array)
print("Sorted array:", sorted_array)