def method():
    # Example list of positive integers
    numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    
    # Create a dictionary to count the frequency of each integer
    frequency = {}
    for num in numbers:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1
    
    # Find the greatest integer that has a frequency greater than or equal to its value
    result = -1
    for num, freq in frequency.items():
        if freq >= num:
            result = max(result, num)
    
    return result

# Test case
output = method()
print(output)  # Expected output: 3