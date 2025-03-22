def method(s):
  output = []
  for char in s:
    if char.isalpha():
      output.append(char.swapcase())
    else:
      output.append(char)
  if not any(char.isalpha() for char in s):
    output.reverse()
  return ''.join(output)

# Test Case
test_string = "Hello5World!"
result = method(test_string)
print(f"Input: {test_string}")
print(f"Output: {result}")