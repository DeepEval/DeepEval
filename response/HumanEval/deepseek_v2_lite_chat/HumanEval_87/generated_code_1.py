import numpy as np

def method():
    # Input: A 2D list (matrix) and an integer x
    # Output: A list of tuples representing the coordinates of x in the matrix
    
    def find_coordinates(matrix, x):
        coordinates = []
        for row_index, row in enumerate(matrix):
            for column_index, element in np.ndenumerate(row):
                if element == x:
                    coordinates.append((row_index, column_index[1]))  # Using column_index[1] for descending order
        return sorted(coordinates, key=lambda coord: (coord[0], -coord[1]))
    
    # Example input
    matrix = np.array([
        [1, 2, 3, 4],
        [5, 6],
        [7, 8, 9],
        [10]
    ])
    
    x = 5
    
    # Find and sort the coordinates of x
    output = find_coordinates(matrix, x)
    return output

# Test case
output = method()
print(output)