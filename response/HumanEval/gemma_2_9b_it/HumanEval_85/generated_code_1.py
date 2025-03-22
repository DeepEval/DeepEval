def method():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    output = 0
    for i in range(1, len(lst), 2):
        if lst[i] % 2 == 0:
            output += lst[i]
    return output

# Test Case
print(method())  # Expected Output: 12 (2 + 6 + 10)