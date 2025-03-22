def method(lst, x):
    # List to store coordinates of the occurrences of x
    coordinates = []

    # Iterate over each row
    for row_index, row in enumerate(lst):
        # Iterate over each element in the row
        for col_index, value in enumerate(row):
            # Check if the current value is equal to x
            if value == x:
                # Append the coordinate as a tuple (row_index, col_index)
                coordinates.append((row_index, col_index))

    # Sort the coordinates by rows in ascending order
    # and by columns in descending order within each row
    coordinates.sort(key=lambda coord: (coord[0], -coord[1]))

    return coordinates

# Test case for validation
lst = [
    [5, 1, 7],
    [2, 9, 5, 3],
    [5, 0],
    [3, 5, 5]
]

x = 5
output = method(lst, x)
print(output)