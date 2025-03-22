import numpy as np

def is_odd(n):
    """Check if the digit is odd."""
    return int(str(n)[0]) % 2 != 0 or int(str(n)[-1]) % 2 != 0

def method(arr):
    """
    Count the number of elements in the array that are greater than 10
    and have odd digits as both first and last.
    """
    # Filter the array to include only numbers greater than 10
    filtered_arr = [num for num in arr if num > 10 and is_odd(num)]
    
    # Return the count of such numbers
    return len(filtered_arr)

# Test case
def test_method():
    test_array = np.array([12, 15, 25, 35, 45, 55, 65, 101, 105, 107, 109])
    # assert method(test_array) == 3, "Test case failed"
    output=method(test_array)
    # print("All test cases passed.")

# Run the test case
test_method()