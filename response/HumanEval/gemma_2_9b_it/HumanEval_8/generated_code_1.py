def method(numbers):
    """
    Calculates the sum and product of integers in a list.

    Args:
        numbers (list): A list of integers.

    Returns:
        tuple: A tuple containing the sum and product of the integers.
               If the list is empty, the sum is 0 and the product is 1.
    """
    sum_of_numbers = 0
    product_of_numbers = 1

    for number in numbers:
        sum_of_numbers += number
        product_of_numbers *= number

    return (sum_of_numbers, product_of_numbers)

# Test Case
numbers = [1, 2, 3, 4, 5]
output = method(numbers)
print(f"Sum: {output[0]}, Product: {output[1]}") 

empty_list = []
output = method(empty_list)
print(f"Sum: {output[0]}, Product: {output[1]}")