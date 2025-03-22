def method(a, b):
  """
  Given two positive integers a and b, return the even digits between a and b, in ascending order. 

  Args:
    a: The starting integer.
    b: The ending integer.

  Returns:
    A list of even digits between a and b in ascending order.
  """
  even_digits = []
  for num in range(a, b + 1):
    for digit in str(num):
      if int(digit) % 2 == 0 and int(digit) not in even_digits:
        even_digits.append(int(digit))
  even_digits.sort()
  return even_digits

# Test Case
a = 10
b = 25
output = method(a, b)
print(f"Even digits between {a} and {b}: {output}")