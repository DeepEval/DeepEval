def method(arr, k):
    # Import the necessary module
    import heapq

    # Use heapq.nlargest to get the k largest elements
    largest_k_elements = heapq.nlargest(k, arr)
    
    # Sort the list of largest elements
    largest_k_elements.sort()
    
    return largest_k_elements

# Test case
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
k = 4
output = method(arr, k)
print(output)  # Expected output: [2, 3, 4, 5]