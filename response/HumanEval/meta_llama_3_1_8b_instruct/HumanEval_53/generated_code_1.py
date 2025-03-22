def method(x, y):
    """
    This function adds two numbers x and y.
    
    Parameters:
    x (int or float): The first number to be added.
    y (int or float): The second number to be added.
    
    Returns:
    int or float: The sum of x and y.
    """
    # Check if inputs are numbers
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError("Both inputs must be numbers.")
    
    # Add the numbers
    output = x + y
    
    return output

# Test case for validation
if __name__ == "__main__":
    try:
        x = 5
        y = 7
        result = method(x, y)
        print(f"The sum of {x} and {y} is: {result}")
    except TypeError as e:
        print(f"Error: {e}")