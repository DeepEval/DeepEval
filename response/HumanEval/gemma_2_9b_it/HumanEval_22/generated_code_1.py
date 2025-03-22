def method():
  input_list = [1, "hello", 3.14, 5, True, 7, "world"]
  integers = [x for x in input_list if isinstance(x, int)]
  return integers

# Test case
output = method()
print(f"Filtered integers: {output}")