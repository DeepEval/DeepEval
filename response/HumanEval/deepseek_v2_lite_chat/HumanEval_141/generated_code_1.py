import re

def is_valid_file_name(filename):
    # Check if there are more than three digits
    if re.search(r'\d{4,}', filename):
        return 'No'
    
    # Check if there is exactly one dot '.'
    if re.search(r'\.', filename) and filename.count('.') != 1:
        return 'No'
    
    # Check if the name starts with a letter and is followed by a dot
    if not re.match(r'^[A-Za-z]\..*', filename) or re.search(r'\.\w', filename):
        return 'No'
    
    # Check if the substring after the dot is one of the specified extensions
    if re.search(r'\..*(\.txt|\.exe|\.dll)$', filename):
        return 'Yes'
    
    return 'No'

# Test case
test_cases = [
    "file.txt",
    "file23.exe",
    "123file.dll",
    "file.",
    "file.txt.pdf",
    "file.txt.dll",
    "file.txt.DLL",
    "file.txt.dll ",
    "file.txt.dll.",
    "file..txt.dll",
    "file.txt.dll..",
    "file.dll.txt",
    "file.dll.txt.exe",
    "file.dll",
    "file.dll.txt",
    "file.txt",
    "file.txt.dll"
]

for test in test_cases:
    print(f"Filename: {test} - Valid: {is_valid_file_name(test)}")