import re

def method(file_name):
  # Create a regular expression to validate the file name
  pattern = re.compile(r'^[a-zA-Z]+(\d{0,3})?\.[txt|exe|dll]$')

  # Check if the file name matches the regular expression
  if pattern.match(file_name):
    return 'Yes'
  else:
    return 'No'

# Test case
file_name = 'file1.txt'
print(method(file_name))