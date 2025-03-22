def method():
    """
    This function calculates the total sum of digits in binary representation of a given positive integer N.

    Args:
    N (int): A positive integer.

    Returns:
    int: The total sum of digits in binary representation of N.
    """
    
    # Input a positive integer
    # N = int(input("Enter a positive integer: "))

    # 修改：
    N = 10

    # Check if the number is positive
    while N <= 0:
        # N = int(input("Please enter a positive integer: "))

        # 修改：
        N = 10

    # Convert the integer to binary and remove the '0b' prefix
    binary = bin(N)[2:]

    # Initialize a variable to store the sum of digits
    total_sum = 0

    # Iterate over each character (digit) in the binary string
    for digit in binary:
        # Add the integer value of the digit to the total sum
        total_sum += int(digit)

    return total_sum

# Test case
output = method()
print(f"The total sum of digits in binary representation is: {output}")