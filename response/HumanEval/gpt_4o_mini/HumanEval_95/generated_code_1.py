def method(input_dict):
    # Check if the dictionary is empty
    if not input_dict:
        return False
    
    # Get all keys of the dictionary
    keys = input_dict.keys()
    
    # Check if all keys are strings and in lowercase
    all_lower = all(isinstance(key, str) and key.islower() for key in keys)
    
    # Check if all keys are strings and in uppercase
    all_upper = all(isinstance(key, str) and key.isupper() for key in input_dict.keys())
    
    # Return True if either condition is met
    return all_lower or all_upper

# Test case
test_dict1 = {'a': 1, 'b': 2, 'c': 3}  # All keys are lower case
test_dict2 = {'A': 1, 'B': 2, 'C': 3}  # All keys are upper case
test_dict3 = {'A': 1, 'b': 2, 'C': 3}  # Mixed case keys
test_dict4 = {}  # Empty dictionary

print(method(test_dict1))  # Output: True
print(method(test_dict2))  # Output: True
print(method(test_dict3))  # Output: False
print(method(test_dict4))  # Output: False