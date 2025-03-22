def method(numbers):
    # Using list comprehension to filter positive numbers
    output = [num for num in numbers if num > 0]
    return output

# Test case for validation
if __name__ == "__main__":
    test_input = [-10, 15, 0, -5, 20, 25, -30]
    expected_output = [15, 20, 25]
    
    result = method(test_input)
    print("Input:", test_input)
    print("Output:", result)
    print("Expected Output:", expected_output)
    print("Test Passed:", result == expected_output)