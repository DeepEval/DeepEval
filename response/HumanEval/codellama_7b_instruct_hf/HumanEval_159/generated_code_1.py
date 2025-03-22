from typing import List

def method(carrots_ate: int, total_carrots: int) -> List[int]:
    """
    This method calculates the number of carrots eaten and left after meals.

    Args:
    carrots_ate: The number of carrots already eaten.
    total_carrots: The total number of carrots available.

    Returns:
    A list containing the number of carrots eaten and left after meals.
    """

    # Calculate the number of carrots left.
    carrots_left = total_carrots - carrots_ate

    # If there are not enough remaining carrots, eat all of them.
    if carrots_left < 0:
        carrots_left = 0

    # Return the number of carrots eaten and left.
    return [carrots_ate + carrots_left, carrots_left]


# Test case.
carrots_ate = 5
total_carrots = 10

output = method(carrots_ate, total_carrots)

print(output)  # Output: [10, 0]