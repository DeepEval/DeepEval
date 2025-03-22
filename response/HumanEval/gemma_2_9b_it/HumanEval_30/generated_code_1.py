def method():
  numbers = [1, -2, 3, -4, 5, -6, 7, -8, 9]
  positive_numbers = [num for num in numbers if num > 0]
  return positive_numbers

output = method()
print(output)  # Output: [1, 3, 5, 7, 9]