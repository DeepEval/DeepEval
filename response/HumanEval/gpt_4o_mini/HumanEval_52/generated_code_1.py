def method(l, t):
    # Check if all numbers in the list are below the threshold
    output = all(num < t for num in l)
    return output

# Test case for validation
if __name__ == "__main__":
    test_list = [1, 2, 3, 4, 5]
    threshold = 6
    result = method(test_list, threshold)
    print(f"All numbers in {test_list} are below {threshold}: {result}")  # Expected output: True

    # Another test case
    test_list = [1, 2, 3, 4, 5]
    threshold = 4
    result = method(test_list, threshold)
    print(f"All numbers in {test_list} are below {threshold}: {result}")  # Expected output: False