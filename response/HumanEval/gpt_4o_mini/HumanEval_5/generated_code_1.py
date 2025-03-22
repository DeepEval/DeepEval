def method(numbers, delimiter):
    # Initialize an empty list to hold the result
    output = []
    
    # Iterate through the input list and insert elements with the delimiter
    for i in range(len(numbers)):
        output.append(numbers[i])  # Append the current number
        if i < len(numbers) - 1:  # Avoid adding delimiter after the last element
            output.append(delimiter)  # Append the delimiter
    
    return output

# Test case for validation
if __name__ == "__main__":
    numbers = [1, 2, 3, 4]
    delimiter = '-'
    result = method(numbers, delimiter)
    print(result)  # Expected output: [1, '-', 2, '-', 3, '-', 4]