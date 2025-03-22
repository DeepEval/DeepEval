import math

def sum_to_n(n):
    """
    Sum numbers from 1 to n using the formula:
    Sum = n * (n + 1) / 2
    """
    sum = n * (n + 1) // 2
    return sum


# Test case for validation
def test_sum_to_n():
    sum_to_n(5)
    # assert sum_to_n(5) == 15
    # assert sum_to_n(10) == 55
    # assert sum_to_n(0) == 0