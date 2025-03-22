import math

def method(numbers):
  output = 0
  for number in numbers:
    output += math.ceil(number) ** 2

  return output

# Test case
numbers = [2.2, 5.6, 8.9]
print(method(numbers))  # Output: 77