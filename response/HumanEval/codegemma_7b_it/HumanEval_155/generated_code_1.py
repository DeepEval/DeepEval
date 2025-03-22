from typing import Tuple

def method(number: int) -> Tuple[int, int]:
    """
    Given an integer, return a tuple that has the number of even and odd digits respectively.
    """
    even_count = 0
    odd_count = 0
    while number > 0:
        digit = number % 10
        if digit % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
        number //= 10
    return even_count, odd_count

# Test case
number = 123456789
even_count, odd_count = method(number)
print(f"Even digits: {even_count}")
print(f"Odd digits: {odd_count}")