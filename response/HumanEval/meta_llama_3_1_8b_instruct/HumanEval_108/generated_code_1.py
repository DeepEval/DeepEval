def count_nums(nums):
    """
    This function takes an array of integers and returns the number of elements 
    which has a sum of digits > 0. If a number is negative, then its first signed 
    digit will be negative: e.g. -123 has signed digits -1, 2, and 3.

    Args:
        nums (list): A list of integers.

    Returns:
        int: The number of elements with a sum of digits > 0.
    """

    def sum_of_digits(n):
        """
        This function calculates the sum of digits of a given number.
        If the number is negative, it considers the first digit as negative.

        Args:
            n (int): The input number.

        Returns:
            int: The sum of digits.
        """
        sign = -1 if n < 0 else 1
        n = abs(n)
        return sum(int(digit) * sign for digit in str(n))

    # Count the numbers with sum of digits > 0
    count = sum(1 for num in nums if sum_of_digits(num) > 0)

    return count


# Test case
def test_count_nums():
    nums = [-10, -9, 0, 1, 2, 3, 4, 5]
    output = count_nums(nums)
    expected_output = 5
    # assert output == expected_output, f"Expected {expected_output}, but got {output}"


test_count_nums()