def method(arr, k):
    """
    Given an array arr of integers and a positive integer k, 
    return a sorted list of length k with the maximum k numbers in arr.

    Parameters:
    arr (list): The input array of integers.
    k (int): The number of maximum elements to return.

    Returns:
    list: A sorted list of length k with the maximum k numbers in arr.
    """
    # First, we sort the array in descending order to get the maximum numbers first
    sorted_arr = sorted(arr, reverse=True)
    
    # Then, we return the first k elements of the sorted array
    return sorted_arr[:k]

# Test case for validation
arr = [1, 23, 12, 9, 30, 2, 50]
k = 3
output = method(arr, k)
print(output)  # Expected output: [50, 30, 23]