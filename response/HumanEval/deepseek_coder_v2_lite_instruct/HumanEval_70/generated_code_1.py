def method(numbers):
    sorted_numbers = sorted(numbers)
    result = []
    left = 0
    right = len(sorted_numbers) - 1
    
    while left <= right:
        if left == right:
            result.append(sorted_numbers[left])
        else:
            result.append(sorted_numbers[left])
            result.append(sorted_numbers[right])
        left += 1
        right -= 1
    
    return result

# Test case
numbers = [5, 3, 8, 1, 2, 7]
output = method(numbers)
print(output)  # Expected output: [1, 8, 2, 7, 3, 5]