def method(numbers):
    """
    Applies a linear transform to a list of numbers,
    mapping the smallest to 0 and the largest to 1.

    Args:
      numbers: A list of numbers (at least two elements).

    Returns:
      A list of transformed numbers.
    """
    min_num = min(numbers)
    max_num = max(numbers)
    range_ = max_num - min_num
    
    output = [(x - min_num) / range_ for x in numbers]
    return output

# Test case
numbers = [1, 5, 2, 8, 3]
transformed_numbers = method(numbers)
print(f"Original list: {numbers}")
print(f"Transformed list: {transformed_numbers}")