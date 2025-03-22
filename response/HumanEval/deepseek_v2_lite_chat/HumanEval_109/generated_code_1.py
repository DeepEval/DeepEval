def method(arr):
    if not arr:  # Check if the array is empty
        return True
    
    N = len(arr)
    for i in range(N):
        if arr[i] > arr[(i + 1) % N]:  # Check for the first pair where the first element is greater than the second
            return False
    
    return True