def method(numbers):
    """
    From a supplied list of numbers (of length at least two) select and return two that are the closest to each other and return them in order (smaller number, larger number). 

    Args:
      numbers: A list of numbers.

    Returns:
      A list containing two numbers, the closest pair in ascending order.
    """
    numbers.sort()
    closest_pair = [numbers[0], numbers[1]]
    min_difference = abs(numbers[1] - numbers[0])

    for i in range(2, len(numbers)):
        difference = abs(numbers[i] - numbers[i-1])
        if difference < min_difference:
            min_difference = difference
            closest_pair = [numbers[i-1], numbers[i]]
    return closest_pair

# Test Case
numbers = [1, 5, 3, 9, 2]
output = method(numbers)
print(f"The closest pair of numbers in {numbers} is: {output}")