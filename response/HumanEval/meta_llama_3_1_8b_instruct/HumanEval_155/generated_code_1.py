def method(n):
    """
    Given an integer, returns a tuple with the number of even and odd digits respectively.

    Args:
        n (int): The input integer.

    Returns:
        tuple: A tuple containing the count of even and odd digits.
    """
    even_count = 0
    odd_count = 0
    
    # Convert the integer to a string to easily iterate over each digit
    for digit in str(n):
        # Convert the digit back to an integer for comparison
        int_digit = int(digit)
        
        # Check if the digit is even or odd
        if int_digit % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    
    # Return the count of even and odd digits as a tuple
    return (even_count, odd_count)


# Test case for validation
if __name__ == "__main__":
    input_number = 123456
    output = method(input_number)
    print(f"Input Number: {input_number}")
    print(f"Even Digit Count: {output[0]}")
    print(f"Odd Digit Count: {output[1]}")