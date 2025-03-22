def convert_to_float(value):
    """Convert a value to a float, handling both '.' and ',' as decimal points."""
    if isinstance(value, str):
        # Replace ',' with '.' for conversion
        value = value.replace(',', '.')
    return float(value)

def compare_one(a, b):
    # Convert both values to float for comparison
    try:
        a_float = convert_to_float(a)
        b_float = convert_to_float(b)
        
        if a_float > b_float:
            return a
        elif b_float > a_float:
            return b
        else:
            return None
    except ValueError:
        # If conversion fails, return None (could be due to invalid strings)
        return None

def method():
    # Example test cases
    output1 = compare_one(1, 2.5)        # Should return 2.5
    output2 = compare_one(1, "2,3")      # Should return "2,3"
    output3 = compare_one("5,1", "6")    # Should return "6"
    output4 = compare_one("1", 1)         # Should return None
    
    # Return a list of outputs for validation
    return output1, output2, output3, output4

# Call the method to get the outputs
outputs = method()
print(outputs)