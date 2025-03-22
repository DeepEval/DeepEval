import numpy as np

def method(lst):
    # Initialize an empty list to store the even elements at odd indices
    output = []
    
    # Iterate through the list using enumerate to get both the index and value
    for index, value in enumerate(lst):
        # Check if the index is odd and the value is even
        if index % 2 != 0 and value % 2 == 0:
            output.append(value)
    
    return output

# Example test case
if __name__ == "__main__":
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = method(test_list)
    print("Output:", result)