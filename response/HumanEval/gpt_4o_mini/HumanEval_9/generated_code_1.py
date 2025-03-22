def method(input_list):
    rolling_max = []
    current_max = float('-inf')  # Start with the smallest possible value

    for number in input_list:
        if number > current_max:
            current_max = number  # Update current maximum if the new number is larger
        rolling_max.append(current_max)  # Append the current maximum to the rolling_max list

    return rolling_max

# Test case to validate the method
if __name__ == "__main__":
    test_input = [3, 1, 4, 1, 5, 9, 2, 6]
    output = method(test_input)
    print(output)  # Expected output: [3, 3, 4, 4, 5, 9, 9, 9]