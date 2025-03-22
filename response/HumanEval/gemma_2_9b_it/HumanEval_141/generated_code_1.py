import re

def method(file_name):
    """
    Validates a file name based on the given criteria.

    Args:
        file_name: A string representing the file name.

    Returns:
        'Yes' if the file name is valid, 'No' otherwise.
    """
    if len(re.findall(r'\d+', file_name)) > 3:
        return 'No'  
    if file_name.count('.') != 1:
        return 'No'
    parts = file_name.split('.')
    if not parts[0].isalpha():
        return 'No'
    if parts[1] not in ['txt', 'exe', 'dll']:
        return 'No'
    return 'Yes'

# Test case
file_name1 = "document.txt"
file_name2 = "123document.txt"
file_name3 = "document.docx"
file_name4 = "abc1.exe"
file_name5 = "12345.exe"

print(f"'{file_name1}' is valid: {method(file_name1)}")
print(f"'{file_name2}' is valid: {method(file_name2)}")
print(f"'{file_name3}' is valid: {method(file_name3)}")
print(f"'{file_name4}' is valid: {method(file_name4)}")
print(f"'{file_name5}' is valid: {method(file_name5)}")