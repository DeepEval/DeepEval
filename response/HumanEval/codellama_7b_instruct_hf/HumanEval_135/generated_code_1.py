# importing necessary packages
import numpy as np

# creating the function
def method():
    """
    This function returns the largest index of an element which is not greater than or equal to the element immediately preceding it. If no such element exists then return -1. The given array will not contain duplicate values.
    """
    
    # getting the array from the user
    # arr = np.array(input("Enter the array: ").split(), dtype=int)
    # 修改：
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # initializing the output variable
    output = -1
    
    # iterating through the array from the second element
    for i in range(1, len(arr)):
        # checking if the current element is greater than or equal to the previous element
        if arr[i] <= arr[i-1]:
            # if yes, then updating the output variable
            output = i
    
    # returning the output variable
    return output

# creating a test case
test_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# calling the method
output = method()

# printing the output
print(output)