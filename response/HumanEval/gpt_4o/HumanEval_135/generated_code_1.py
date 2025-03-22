def method(arr):
    # Initialize the variable to hold the largest index
    index = -1

    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # Check if the current element is not greater than or equal to the previous element
        if arr[i] < arr[i - 1]:
            index = i
    
    return index

# Test the method with a test case
test_array = [1, 2, 5, 4, 6, 7, 10]
output = method(test_array)
print(output)  # This should print 3 because arr[3] (4) is not greater than arr[2] (5)