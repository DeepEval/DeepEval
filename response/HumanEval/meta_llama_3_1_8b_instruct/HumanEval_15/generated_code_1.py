def method(n):
    """
    Returns a string containing space-delimited numbers starting from 0 up to n inclusive.

    Args:
        n (int): The upper limit of the range of numbers to be included in the string.

    Returns:
        str: A string containing space-delimited numbers from 0 to n inclusive.
    """
    return''.join(map(str, range(n + 1)))

# Test case for validation
if __name__ == "__main__":
    n = 10
    output = method(n)
    print(f"Input: n = {n}")
    print(f"Output: {output}")
    print(f"Expected Output: 0 1 2 3 4 5 6 7 8 9 10")
    # assert output == '0 1 2 3 4 5 6 7 8 9 10', f"Expected output '0 1 2 3 4 5 6 7 8 9 10' but got '{output}'"
    # print("Test case passed!")