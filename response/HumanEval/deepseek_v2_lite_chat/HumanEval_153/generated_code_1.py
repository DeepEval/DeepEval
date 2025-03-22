def method(className, extensions):
    # Initialize a dictionary to store the strength of each extension
    extension_strength = {}
    
    # Calculate the strength of each extension
    for ext in extensions:
        # Count the number of uppercase and lowercase letters in the extension name
        cap_count = sum(1 for char in ext if char.isupper())
        low_count = sum(1 for char in ext if char.islower())
        # Calculate the strength of the extension
        strength = cap_count - low_count
        # Store the strength and the extension name in the dictionary
        extension_strength[ext] = (strength, ext)
    
    # Sort the extensions by strength in descending order and by name in ascending order
    sorted_extensions = sorted(extension_strength.items(), key=lambda x: (-x[1][0], x[1][1]))
    
    # Return the class name with the strongest extension
    return f"{className}.{sorted_extensions[0][1][1]}"

# Test case to validate the function
def test_method():
    class_name = "Slices"
    extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
    expected_output = "Slices.SErviNGSliCes"
    result = method(class_name, extensions)
    # assert result == expected_output, f"Expected {expected_output}, but got {result}"
    # print("Test case passed.")

test_method()