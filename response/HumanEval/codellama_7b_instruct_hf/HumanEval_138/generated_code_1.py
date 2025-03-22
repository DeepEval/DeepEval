from itertools import combinations_with_replacement

def method(n):
    """
    Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers

    Args:
        n: The number to be evaluated.

    Returns:
        True if n can be written as the sum of exactly 4 positive even numbers, False otherwise.
    """

    # Check if n is a positive even number
    if n % 2 != 0:
        return False

    # Generate all possible combinations of 4 positive even numbers
    combinations = combinations_with_replacement(range(2, n + 1, 2), 4)

    # Check if any combination sums to n
    for combination in combinations:
        if sum(combination) == n:
            return True

    # No combination sums to n
    return False

# Test case
test_case = 12
result = method(test_case)
print(f"Can {test_case} be written as the sum of exactly 4 positive even numbers? {result}")