def method(text):
  """
  Encodes a string by cycling groups of three characters.

  Args:
    text: The string to encode.

  Returns:
    The encoded string.
  """

  output = ''
  for i in range(0, len(text), 3):
    group = text[i:i+3]
    if len(group) < 3:
      group += ' ' * (3 - len(group)) 
    output += group[::-1]
  return output

# Test Case
test_string = "This is a test string"
encoded_string = method(test_string)
print(f"Original String: {test_string}")
print(f"Encoded String: {encoded_string}")