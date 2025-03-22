def method():
    x = [12, 345, 2, 6, 7896]
    output = sorted([num for num in x if all(digit % 2 for digit in str(num))])
    return output

# Test Case
print(method())  # Expected Output: [123, 789]