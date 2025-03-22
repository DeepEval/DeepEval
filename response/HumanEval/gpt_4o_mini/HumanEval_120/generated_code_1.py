import heapq

def method(arr, k):
    # Use a min-heap to keep track of the top k elements
    if k <= 0 or k > len(arr):
        return []
    
    # Create a min-heap with the first k elements
    min_heap = arr[:k]
    heapq.heapify(min_heap)

    # Iterate over the rest of the array
    for num in arr[k:]:
        if num > min_heap[0]:  # Compare with the smallest in the heap
            heapq.heappop(min_heap)  # Remove the smallest
            heapq.heappush(min_heap, num)  # Add the new number

    # Convert the heap to a sorted list before returning
    output = sorted(min_heap)
    return output

# Test case
arr = [3, 1, 5, 12, 2, 11]
k = 3
print(method(arr, k))  # Expected output: [5, 11, 12]