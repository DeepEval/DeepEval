def method():
  """
  Calculates the Brazilian factorial of a given number.

  Returns:
      int: The Brazilian factorial of the input number.
  """
  # n = int(input("Enter a positive integer: "))

  # 修改为固定值
  n = 5
  
  brazilian_factorial = 1
  for i in range(1, n + 1):
    factorial = 1
    for j in range(1, i + 1):
      factorial *= j
    brazilian_factorial *= factorial
  return brazilian_factorial

# Test case
result = method()
print(f"The Brazilian factorial of the input number is: {result}")