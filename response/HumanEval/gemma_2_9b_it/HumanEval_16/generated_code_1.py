def method():
  # input_string = input("Enter a string: ")

  # 修改为固定值
  input_string = "Hello, World!"
  
  distinct_chars = set(input_string.lower())  
  output = len(distinct_chars)
  return output

# Test Case
result = method()
print(f"The string has {result} distinct characters.")