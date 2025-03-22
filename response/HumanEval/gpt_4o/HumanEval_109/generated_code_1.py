def method(arr):
    if not arr:
        return True
    
    N = len(arr)
    # Concatenate the array with itself
    doubled_arr = arr + arr
    
    # Check every possible rotation (window of size N in doubled_arr)
    for i in range(N):
        # Extract a window of size N
        window = doubled_arr[i:i + N]
        # Check if this window is sorted
        if window == sorted(window):
            return True
    
    return False

# Test case
test_arr = [3, 4, 5, 1, 2]
print(method(test_arr))  # Output should be True because [1, 2, 3, 4, 5] can be obtained by right shifting