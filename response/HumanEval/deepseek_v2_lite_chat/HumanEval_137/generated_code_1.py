import re

def compare_one(a, b):
    # Check if both are integers or both are floats
    if isinstance(a, int) and isinstance(b, int):
        return a if a > b else b
    elif isinstance(a, float) and isinstance(b, float):
        return a if a > b else b
    
    # Check if both are strings
    if isinstance(a, str) and isinstance(b, str):
        # Check if the first is an integer and the second is a float
        if isinstance(a, int) and isinstance(b, float):
            return str(a)
        # Check if the first is a float and the second is an integer
        elif isinstance(a, float) and isinstance(b, int):
            return str(b)
        # Check if both are floats
        elif isinstance(a, float) and isinstance(b, float):
            # Compare the strings after stripping commas
            return b if re.sub(r'\D', '', a) > re.sub(r'\D', '', b) else a
        # Check if both are integers
        elif isinstance(a, int) and isinstance(b, int):
            return str(a)
    
    # If the comparison fails, return None
    return None

# Test case for validation
print(compare_one(1, 2.5))  # ➞ 2.5
print(compare_one(1, "2,3"))  # ➞ "2,3"
print(compare_one("5,1", "6"))  # ➞ "6"
print(compare_one("1", 1))  # ➞ None