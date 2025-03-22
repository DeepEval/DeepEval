def compare_one(val1, val2):
    def parse_value(val):
        if isinstance(val, str):
            # Convert ',' to '.' to handle European decimal format
            val = val.replace(',', '.')
            return float(val)
        return float(val)
    
    parsed_val1 = parse_value(val1)
    parsed_val2 = parse_value(val2)
    
    if parsed_val1 > parsed_val2:
        return val1
    elif parsed_val2 > parsed_val1:
        return val2
    else:
        return None

def method():
    # Example test cases
    test_cases = [
        (1, 2.5),
        (1, "2,3"),
        ("5,1", "6"),
        ("1", 1),
        ("3.5", 3.5),
        ("4,0", "4.0")
    ]
    
    output = [compare_one(val1, val2) for val1, val2 in test_cases]
    return output

# Run the method to see the results
print(method())