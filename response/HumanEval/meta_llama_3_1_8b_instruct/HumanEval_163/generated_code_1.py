def method(a, b):
    """
    This function takes two positive integers a and b as input.
    It returns a list of even digits between a and b in ascending order.

    Parameters:
    a (int): The lower bound of the range (inclusive).
    b (int): The upper bound of the range (inclusive).

    Returns:
    list: A list of even digits between a and b in ascending order.
    """

    # Initialize an empty list to store even digits
    even_digits = []

    # Iterate over the range from a to b (inclusive)
    for num in range(a, b + 1):
        # Convert the number to a string to easily iterate over its digits
        str_num = str(num)
        
        # Iterate over each digit in the number
        for digit in str_num:
            # Convert the digit back to an integer to check if it's even
            int_digit = int(digit)
            
            # Check if the digit is even
            if int_digit % 2 == 0:
                # If the digit is even, add it to the list
                even_digits.append(int_digit)

    # Remove duplicates by converting the list to a set and then back to a list
    even_digits = list(set(even_digits))

    # Sort the list in ascending order
    even_digits.sort()

    # Return the list of even digits
    return even_digits


# Test case
output = method(10, 50)
print(output)  # Expected output: [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]