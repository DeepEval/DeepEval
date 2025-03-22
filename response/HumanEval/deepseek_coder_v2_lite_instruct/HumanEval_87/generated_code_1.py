def method():
    lst = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9],
        [10]
    ]
    x = 5
    output = []

    # Iterate through the list to find the coordinates of x
    for i, row in enumerate(lst):
        for j, value in enumerate(row):
            if value == x:
                output.append((i, j))

    # Sort the coordinates by rows in ascending order and by columns in descending order
    output.sort(key=lambda coord: (coord[0], -coord[1]))

    return output

# Test case
print(method())  # Expected output: [(1, 0)]