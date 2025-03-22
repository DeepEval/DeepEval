def method(fruit_str, total_fruits):
    """
    Calculate the number of mango fruits in a basket given the total number of apples and oranges and the total number of fruits.

    Args:
    fruit_str (str): A string that represents the total number of apples and oranges in the basket. The format is 'apples:oranges'.
    total_fruits (int): The total number of fruits in the basket.

    Returns:
    int: The number of mango fruits in the basket.
    """
    
    # Split the string to get the number of apples and oranges
    apples, oranges = map(int, fruit_str.split(':'))
    
    # Calculate the total number of apples and oranges
    total_apples_oranges = apples + oranges
    
    # Calculate the number of mango fruits
    mango_fruits = total_fruits - total_apples_oranges
    
    return mango_fruits


# Test case
print(method('20:30', 60))  # Output: 10