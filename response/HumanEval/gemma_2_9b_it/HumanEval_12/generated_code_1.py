def method():
  strings = ["apple", "banana", "cherry", "date"] 
  longest_string = None
  max_length = 0
  for string in strings:
    if len(string) > max_length:
      max_length = len(string)
      longest_string = string
  return longest_string

output = method()
print(f"The longest string is: {output}")