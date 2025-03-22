import heapq

def method(arr, k):
    # Find the k largest elements using heapq
    k_largest = heapq.nlargest(k, arr)
    
    # Sort the result before returning
    output = sorted(k_largest)
    
    return output

# Test case
arr = [3, 1, 5, 12, 2, 11, 7, 6]
k = 3
print(method(arr, k))  # Expected output: [11, 12, 7]