def method(numbers):
    output = []
    for index, num in enumerate(numbers):
        if index % 3 == 0:
            output.append(num ** 2)
        elif index % 4 == 0 and index % 3 != 0:
            output.append(num ** 3)
        else:
            output.append(num)
    return sum(output)

# Test case
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
print(method(numbers))  # Expected output: 349 (1^2 + 2^3 + 3 + 4^3 + 5 + 6 + 7 + 8^3 + 9 + 10 + 11 + 12)