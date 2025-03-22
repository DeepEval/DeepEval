import heapq

def method(arr, k):
  """
  Given an array arr of integers and a positive integer k, return a sorted list of length k with the maximum k numbers in arr.

  Args:
    arr: A list of integers.
    k: A positive integer.

  Returns:
    A sorted list of length k with the maximum k numbers in arr.
  """
  return heapq.nlargest(k, arr)

# Test Case
arr = [3, 2, 1, 5, 6, 4]
k = 2
output = method(arr, k)
print(f"Maximum {k} numbers in {arr} are: {output}")