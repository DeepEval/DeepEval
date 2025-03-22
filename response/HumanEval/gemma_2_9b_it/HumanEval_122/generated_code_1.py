def method(arr, k):
    """
    Given a non-empty array of integers arr and an integer k, return the sum of the elements with at most two digits from the first k elements of arr. 
    """
    output = 0
    for i in range(k):
      if 10 >= arr[i] >= 0:
        output += arr[i]
      elif 99 >= arr[i] >= 10:
        output += arr[i]
    return output

# Test Case
arr = [1, 23, 4, 55, 67, 89, 90]
k = 3
output = method(arr, k)
print(f"Sum of elements with at most two digits from the first {k} elements of {arr}: {output}")