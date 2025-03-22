from heapq import nlargest

def method(arr, k):
    """
    :param arr: list of integers
    :param k: positive integer
    :return: list of length k with the maximum k numbers in arr
    """
    return nlargest(k, arr)

# Test case
arr = [3, 2, 1, 5, 6, 4]
k = 3

# Output
output = method(arr, k)

print(output) # Output: [6, 5, 4]