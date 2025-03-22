def method():
    input_strings = ["12345", "67890", "13579", "2468"]
    output = []
    for string in input_strings:
        odd_count = 0
        for digit in string:
            if int(digit) % 2 != 0:
                odd_count += 1
        output.append(odd_count)
    return output

# Test Case
result = method()
print(result)  # Expected Output: [5, 0, 5, 0]