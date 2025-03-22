def compare_one(*args):
    """
    Compare two variables of any type (int, float, string) representing real numbers.
    Return the larger variable in its given variable type. Return None if the values are equal.
    
    Args:
    *args: Variable number of arguments of type int, float, or string.
    
    Returns:
    The larger variable in its given variable type, or None if the values are equal.
    """
    
    # Initialize max_val to None
    max_val = None
    
    # Iterate over each input
    for arg in args:
        
        # Check if max_val is None (i.e., we haven't compared any inputs yet)
        if max_val is None:
            
            # If arg is a string, try to convert it to a float
            if isinstance(arg, str):
                try:
                    arg = float(arg.replace(",", "."))
                except ValueError:
                    # If the conversion fails, skip this input
                    continue
            
            # Update max_val to the current arg
            max_val = arg
        
        # If arg is a string, try to convert it to a float
        elif isinstance(arg, str):
            try:
                arg = float(arg.replace(",", "."))
            except ValueError:
                # If the conversion fails, skip this input
                continue
        
        # Compare the current max_val with the current arg
        elif isinstance(arg, int):
            if arg > max_val:
                max_val = arg
        elif isinstance(arg, float):
            if arg > max_val:
                max_val = arg
    
    # Return the max_val, or None if it's still None (i.e., all inputs were equal)
    return max_val

# Test cases
print(compare_one(1, 2.5))  # ➞ 2.5
print(compare_one(1, "2,3"))  # ➞ "2,3"
print(compare_one("5,1", "6"))  # ➞ "6"
print(compare_one("1", 1))  # ➞ None