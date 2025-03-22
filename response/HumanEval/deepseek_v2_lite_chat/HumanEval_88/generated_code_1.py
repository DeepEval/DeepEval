import numpy as np

def method(arr):
    # Create a copy of the input array
    sorted_arr = arr.copy()
    
    # Sort the array based on the condition
    sorted_arr.sort(key=lambda x: (arr[0] + arr[-1]) % 2)
    
    # Return the sorted array
    return sorted_arr

# Example usage:
# arr = np.array([1, 2, 3, 4, 5])
# sorted_arr = method(arr)
# print(sorted_arr)  # This line would be executed in a complete program context

# Example test case (you can uncomment this line and run the code to validate the function)
# arr = np.array([4, 5, 6, 7, 8])
# sorted_arr = method(arr)
# print(sorted_arr)  # Example output