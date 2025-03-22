def method(file_name):
    import re
    
    # Check for exactly one dot in the file name
    if file_name.count('.') != 1:
        return 'No'
    
    # Split the file name into the base name and extension
    base, ext = file_name.split('.')
    
    # Check if the extension is valid
    if ext not in ['txt', 'exe', 'dll']:
        return 'No'
    
    # Check if the base name is not empty and starts with a letter
    if not base or not re.match(r'^[A-Za-z]', base):
        return 'No'
    
    # Count the number of digits in the base name
    digits_count = sum(c.isdigit() for c in base)
    
    # Check if there are more than three digits
    if digits_count > 3:
        return 'No'
    
    return 'Yes'

# Test cases
print(method("example1.txt"))  # Should return 'Yes'
print(method("file123.exe"))   # Should return 'Yes'
print(method("doc4567.dll"))   # Should return 'No' (more than 3 digits)
print(method(".hidden.txt"))   # Should return 'No' (base name is empty)
print(method("invalidfile.doc")) # Should return 'No' (invalid extension)
print(method("2start.txt"))     # Should return 'No' (doesn't start with a letter)