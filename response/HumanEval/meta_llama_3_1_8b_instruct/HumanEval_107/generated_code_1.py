import math

def count_palindromes(n):
    """
    Counts the number of even and odd integer palindromes in the range (1, n).
    
    Args:
    n (int): A positive integer.
    
    Returns:
    tuple: A tuple containing the count of even and odd integer palindromes.
    """
    
    # Initialize counters for even and odd palindromes
    even_palindromes = 0
    odd_palindromes = 0
    
    # Iterate over the range from 1 to n (inclusive)
    for i in range(1, n + 1):
        # Convert the integer to a string to check if it's a palindrome
        str_i = str(i)
        
        # Check if the string is equal to its reverse
        if str_i == str_i[::-1]:
            # If it's a palindrome, check if it's even or odd
            if i % 2 == 0:
                # If it's even, increment the even palindrome counter
                even_palindromes += 1
            else:
                # If it's odd, increment the odd palindrome counter
                odd_palindromes += 1
    
    # Return a tuple containing the count of even and odd palindromes
    return (even_palindromes, odd_palindromes)


def method():
    """
    Returns a tuple containing the count of even and odd integer palindromes in the range (1, n).
    
    Args:
    None
    
    Returns:
    tuple: A tuple containing the count of even and odd integer palindromes.
    """
    
    # Test case: count palindromes up to 100
    n = 100
    output = count_palindromes(n)
    return output


# Test the function
output = method()
print(output)