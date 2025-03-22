def method(hex_string):
    """
    This function counts the number of hexadecimal digits in a given string that are prime numbers.

    Args:
        hex_string (str): A string of hexadecimal numbers.

    Returns:
        int: The number of prime hexadecimal digits in the input string.
    """

    # Define a helper function to check if a number is prime
    def is_prime(n):
        """
        This function checks if a number is prime.

        Args:
            n (int): The number to check.

        Returns:
            bool: True if the number is prime, False otherwise.
        """
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    # Define a dictionary to map hexadecimal digits to their decimal equivalents
    hex_to_decimal = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15
    }

    # Initialize a counter for prime hexadecimal digits
    prime_count = 0

    # Iterate over each character in the input string
    for char in hex_string:
        # Check if the character is a hexadecimal digit
        if char in hex_to_decimal:
            # Check if the decimal equivalent of the hexadecimal digit is prime
            if is_prime(hex_to_decimal[char]):
                # Increment the prime count if it is prime
                prime_count += 1

    # Return the count of prime hexadecimal digits
    return prime_count


# Test case
print(method("2B3D"))  # Output: 3