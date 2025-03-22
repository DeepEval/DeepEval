def method(lst, x):
    output = []
    for row_index, row in enumerate(lst):
        for col_index, value in enumerate(row):
            if value == x:
                output.append((row_index, col_index))
    output.sort(key=lambda item: (item[0], -item[1]))  # Sort by row, then descending column
    return output

# Test Case
lst = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 10, 11], [12, 13, x=4]]
x = 10
result = method(lst, x)
print(f"Coordinates of {x} in the list: {result}")