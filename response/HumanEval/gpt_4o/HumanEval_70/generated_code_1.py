def method(arr):
    # Sort the array first
    arr.sort()
    
    # Initialize pointers for the beginning and end of the array
    left = 0
    right = len(arr) - 1
    
    # Prepare a list to hold the result
    result = []
    
    # Use a flag to toggle between selecting the minimum or maximum
    select_min = True
    
    # Iterate until the two pointers meet or cross
    while left <= right:
        if select_min:
            result.append(arr[left])
            left += 1
        else:
            result.append(arr[right])
            right -= 1
        
        # Toggle the selection mode
        select_min = not select_min
    
    return result

# Test case
input_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
output = method(input_list)
print(output)  # Expected output: [1, 9, 1, 6, 2, 5, 3, 5, 3, 4]