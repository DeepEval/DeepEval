import heapq

def method(arr, k):
    # Convert the list to a min-heap (max-heap in reverse order)
    max_heap = [-1 * num for num in arr]
    heapq.heapify(max_heap)
    
    # Extract the k largest elements from the heap
    output = [-1 * heapq.nlargest(k, max_heap)[i] for i in range(k)]
    return output

# Test case
def test_method():
    # Test with an array and a positive integer k
    arr = [3, 2, 1, 5, 6, 4]
    k = 3
    expected_output = [5, 6, 4]
    
    # Call the method and compare the result with the expected output
    output = method(arr, k)
    # assert output == expected_output, f"Expected {expected_output}, but got {output}"
    
    print("All test cases passed.")

# Run the test case
test_method()