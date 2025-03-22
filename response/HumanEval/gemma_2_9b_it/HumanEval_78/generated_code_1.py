def method(hex_str):
    """
    Counts the number of prime hexadecimal digits in a given string.

    Args:
      hex_str: A string representing a hexadecimal number.

    Returns:
      The number of prime hexadecimal digits in the input string.
    """

    prime_digits = ['2', '3', '5', '7', 'B', 'D']
    count = 0
    for digit in hex_str:
        if digit in prime_digits:
            count += 1
    return count


# Test case
hex_number = "A2B3C5F7"
prime_count = method(hex_number)
print(f"Number of prime hexadecimal digits in '{hex_number}': {prime_count}")