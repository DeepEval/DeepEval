def method():
  input_list = ["apple", "banana", "cherry", "date", "elderberry"]
  output = []
  for word in input_list:
    if len(word) % 2 == 0:
      output.append(word)
  output.sort(key=lambda word: (len(word), word)) 
  return output

# Test case
print(method())