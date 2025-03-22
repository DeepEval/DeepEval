def method(input_dict):
    if not input_dict:  # Check if the dictionary is empty
        return False

    keys = list(input_dict.keys())
    
    # Check if all keys are lowercase
    all_lower = all(isinstance(key, str) and key.islower() for key in keys)
    
    # Check if all keys are uppercase
    all_upper = all(isinstance(key, str) and key.isupper() for key in keys)
    
    return all_lower or all_upper

# Test case for validation
test_case_1 = {"apple": 1, "banana": 2, "cherry": 3}  # Should return True, all keys are lowercase
test_case_2 = {"APPLE": 1, "BANANA": 2, "CHERRY": 3}  # Should return True, all keys are uppercase
test_case_3 = {"Apple": 1, "Banana": 2, "Cherry": 3}  # Should return False, mixed case
test_case_4 = {}  # Should return False, empty dictionary

print(method(test_case_1))  # Output: True
print(method(test_case_2))  # Output: True
print(method(test_case_3))  # Output: False
print(method(test_case_4))  # Output: False