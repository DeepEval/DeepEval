def compare_one(a, b):
    if isinstance(a, str) and isinstance(b, str):
        a_float = float(a.replace(',', '.'))
        b_float = float(b.replace(',', '.'))
    elif isinstance(a, str):
        a_float = float(a.replace(',', '.'))
        b_float = float(b)
    elif isinstance(b, str):
        a_float = float(a)
        b_float = float(b.replace(',', '.'))
    else:
        a_float = a
        b_float = b

    if a_float > b_float:
        return type(a)(a_float)
    elif b_float > a_float:
        return type(b)(b_float)
    else:
        return None

def method():
    test_cases = [
        (1, 2.5),
        (1, "2,3"),
        ("5,1", "6"),
        ("1", 1),
        (3.14, "3,14"),
        ("10.5", 10.5),
        (10, "10"),
        ("10", 10),
        (1, 1)
    ]
    
    results = []
    for a, b in test_cases:
        result = compare_one(a, b)
        results.append(result)
    
    return results

# Test the method
output = method()
print(output)