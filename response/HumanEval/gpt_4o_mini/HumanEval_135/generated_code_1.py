def method(arr):
    # Initialize the variable to store the largest index
    largest_index = -1
    
    # Loop through the array starting from the second element
    for i in range(1, len(arr)):
        # Check if the current element is less than the previous one
        if arr[i] < arr[i - 1]:
            largest_index = i  # Update the largest index

    return largest_index

# Test case
test_array = [5, 10, 8, 7, 15, 20]
output = method(test_array)
print(output)  # Expected output is 3 (element 7 is less than 8 which is its predecessor)