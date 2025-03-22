def method(lst, x):
    # Initialize an empty list to store the coordinates
    coordinates = []
    
    # Iterate over each row and its index
    for row_index, row in enumerate(lst):
        # Iterate over each column and its index in the current row
        for col_index, value in enumerate(row):
            # Check if the current value matches x
            if value == x:
                # Append the coordinate (row_index, col_index) to the list
                coordinates.append((row_index, col_index))
    
    # Sort the coordinates by row (ascending) and column (descending)
    coordinates.sort(key=lambda coord: (coord[0], -coord[1]))
    
    return coordinates

# Test case for validation
test_lst = [
    [1, 2, 3],
    [4, 5, 2],
    [6, 2, 8, 9],
    [2, 10]
]
x = 2

output = method(test_lst, x)
print(output)  # Expected output: [(0, 1), (1, 2), (2, 1), (3, 0)]