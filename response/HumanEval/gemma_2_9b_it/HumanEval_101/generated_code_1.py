def method():
  input_string = "This, is a test string, with commas, and spaces."
  words = input_string.split(", ")
  words = [word.strip() for word in words]  
  return words

output = method()
print(output)