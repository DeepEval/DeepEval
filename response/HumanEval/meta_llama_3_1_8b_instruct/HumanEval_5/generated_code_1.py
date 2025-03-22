def insert_delimeter(numbers, delimeter):
    """
    Inserts a delimeter between every two consecutive elements of input list.
    
    Parameters:
    numbers (list): A list of numbers.
    delimeter (int or str): The number or string to be inserted between numbers.
    
    Returns:
    list: A new list with the delimeter inserted between each pair of numbers.
    """
    if not isinstance(numbers, list) or not all(isinstance(num, (int, float)) for num in numbers):
        raise ValueError("Input must be a list of numbers.")
    if not isinstance(delimeter, (int, float, str)):
        raise ValueError("Delimeter must be a number or a string.")
    
    return [num for num in numbers] + [delimeter for _ in range(len(numbers) - 1)]

# Test case
numbers = [1, 2, 3, 4, 5]
delimeter = 0
output = insert_delimeter(numbers, delimeter)

print(f"Input List: {numbers}")
print(f"Delimeter: {delimeter}")
print(f"Output List: {output}")