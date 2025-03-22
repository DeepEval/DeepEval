def product_of_odd_digits(n):
    """
    Calculate the product of odd digits in a given positive integer.
    
    Args:
        n (int): A positive integer.
    
    Returns:
        int: The product of the odd digits. Return 0 if all digits are even.
    """
    product = 1
    for digit in str(n):
        # Convert the digit back to an integer to perform multiplication
        digit = int(digit)
        if digit % 2!= 0:
            # If the digit is odd, multiply it with the product
            product *= digit
    return product