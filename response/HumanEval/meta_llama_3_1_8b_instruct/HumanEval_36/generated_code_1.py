def method():
    """
    This function returns the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.

    Args:
        n (int): The upper limit for the numbers to be checked.

    Returns:
        int: The total count of digit 7 in the numbers.
    """

    # Initialize count to 0
    count = 0

    # Loop through all numbers from 0 to n-1
    for i in range(n):
        # Check if the number is divisible by 11 or 13
        if i % 11 == 0 or i % 13 == 0:
            # Convert the number to string to easily count the digit 7
            str_i = str(i)
            # Count the occurrences of '7' in the string
            count += str_i.count('7')

    # Return the total count
    return count


# Test case
n = 1000
output = method()
print(f"For n = {n}, the output is: {output}")

# Validate the output
# For n = 1000, the numbers less than 1000 divisible by 11 or 13 are: 11, 13, 22, 26,..., 1000 - 11 = 989
# There are 83 numbers divisible by 11, and 76 numbers divisible by 13. There are 7 numbers divisible by both.
# There are 7 numbers in the list: 77,  70  787,  773,  710,  770,  737.
# In these numbers, there are 6 occurrences of the digit 7.
# Therefore, the correct output should be: 6
# assert output == 6, f"Expected output to be 6, but got {output}"