def method(a):
  """
  Checks if an integer is a cube of some integer.

  Args:
    a: The integer to check.

  Returns:
    True if a is a cube, False otherwise.
  """
  return round(a**(1/3))**3 == a

# Test case
a = 27
output = method(a)
print(f"Is {a} a cube? {output}")