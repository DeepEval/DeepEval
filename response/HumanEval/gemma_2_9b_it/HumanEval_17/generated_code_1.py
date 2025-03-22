def method():
    input_string = "o| .| o| .|.| .| o" 
    output = []
    for char in input_string:
        if char == 'o':
            output.append(4)
        elif char == 'o|':
            output.append(2)
        elif char == '.|':
            output.append(1)
    return output

# Test Case
test_output = method()
print(test_output)  # Expected Output: [4, 2, 4, 2, 1, 1, 4]