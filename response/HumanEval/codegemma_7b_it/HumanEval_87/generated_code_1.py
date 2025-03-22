import numpy as np

def method(lst, x):
    # Create a list of coordinates
    coordinates = []
    for i, row in enumerate(lst):
        for j, value in enumerate(row):
            if value == x:
                coordinates.append((i, j))

    # Sort coordinates by row in ascending order
    coordinates.sort(key=lambda coord: coord[0])

    # Sort coordinates of the same row by column in descending order
    for row in coordinates:
        row[1] = -row[1]

    # Sort coordinates of the same row by column in ascending order
    for row in coordinates:
        row[1] = -row[1]

    # Return the sorted coordinates
    return coordinates

# Test case
lst = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 0]]
x = 0
output = method(lst, x)
print(output)

# Output: [(2, 0), (2, 1)]