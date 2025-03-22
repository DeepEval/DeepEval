def method():
  numbers = [1.2, 2.7, 3.8, 4.1, 5.6]
  output = sum([int(x) ** 2 for x in numbers])
  return output

# Test case
result = method()
print(f"Sum of squared rounded numbers: {result}")