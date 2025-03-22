import re

def method(file_name):
    # Check for the presence of exactly one dot
    if file_name.count('.') != 1:
        return 'No'
    
    # Split the file name into two parts: before and after the dot
    name, extension = file_name.split('.')
    
    # Check that the substring before the dot is not empty and starts with a letter
    if len(name) == 0 or not name[0].isalpha():
        return 'No'
    
    # Check that the substring after the dot is valid
    valid_extensions = ['txt', 'exe', 'dll']
    if extension not in valid_extensions:
        return 'No'
    
    # Check for the number of digits in the file name
    digit_count = sum(c.isdigit() for c in file_name)
    if digit_count > 3:
        return 'No'
    
    # If all checks passed, the file name is valid
    return 'Yes'

# Test cases
print(method("file123.txt"))  # Should return 'Yes'
print(method("1file.txt"))    # Should return 'No' (starts with a digit)
print(method("file.name.txt")) # Should return 'No' (contains more than one dot)
print(method("file.txt123"))   # Should return 'No' (contains more than 3 digits)
print(method(".hiddenfile.exe")) # Should return 'No' (substring before dot is empty)
print(method("file.dll"))      # Should return 'Yes'