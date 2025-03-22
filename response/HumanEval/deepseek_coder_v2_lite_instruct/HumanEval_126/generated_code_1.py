def method():
    numbers = [1, 2, 2, 4, 5]
    if len(numbers) == 0 or len(numbers) == 1:
        return True
    for i in range(len(numbers) - 1):
        if numbers[i] >= numbers[i + 1]:
            return False
        if numbers[i] == numbers[i + 1]:
            return False
    return True

# Test case
print(method())  # Expected output: False