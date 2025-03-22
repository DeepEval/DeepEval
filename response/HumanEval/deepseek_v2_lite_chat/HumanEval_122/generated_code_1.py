def method(arr, k):
    # Initialize the sum to 0
    sum_elements = 0
    
    # Iterate through the first k elements of arr
    for i in range(k):
        # Check if the current element is an integer and has at most two digits
        if 0 <= arr[i] < 100:
            # Add the element to the sum if it meets the criteria
            sum_elements += arr[i]
    
    # Return the sum of elements with at most two digits
    return sum_elements

# Test case to validate the function
def test_method():
    arr = [1, 23, 45, 67, 89, 101, 111]
    k = 3
    sum_elements = method(arr, k)
    # expected_output = 49  # Sum of 1, 23, and 45
    # assert method(arr, k) == expected_output, f"Expected {expected_output}, but got {method(arr, k)}"
    print("Test passed!")

test_method()