import re

def method(file_name):
    """
    This function checks if a file's name is valid.
    
    A file's name is considered to be valid if and only if all the following conditions are met:
    - There should not be more than three digits ('0'-'9') in the file's name.
    - The file's name contains exactly one dot '.' 
    - The substring before the dot should not be empty, and it starts with a letter from the latin alphapet ('a'-'z' and 'A'-'Z').
    - The substring after the dot should be one of these: ['txt', 'exe', 'dll']

    Args:
        file_name (str): The name of the file to be checked.

    Returns:
        str: 'Yes' if the file's name is valid, 'No' otherwise.
    """

    # Check if the file's name contains exactly one dot
    if file_name.count('.')!= 1:
        return 'No'

    # Split the file's name into two substrings
    name, extension = file_name.rsplit('.', 1)

    # Check if the substring before the dot starts with a letter from the latin alphapet
    if not re.match('^[a-zA-Z]', name):
        return 'No'

    # Check if the substring before the dot is not empty
    if not name:
        return 'No'

    # Check if there are more than three digits in the file's name
    if len(re.findall('\d', file_name)) > 3:
        return 'No'

    # Check if the substring after the dot is valid
    if extension not in ['txt', 'exe', 'dll']:
        return 'No'

    # If all conditions are met, return 'Yes'
    return 'Yes'

# Test case
print(method('abc.txt'))  # Expected output: Yes
print(method('abc.txt.exe'))  # Expected output: No
print(method('123.txt'))  # Expected output: No
print(method('a.txt'))  # Expected output: No
print(method('.txt'))  # Expected output: No
print(method('abc.exe123'))  # Expected output: No