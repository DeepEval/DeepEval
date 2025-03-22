def method():
  operator = ["+", "-", "*", "//", "**"]
  operand = [2, 3, 4, 5, 6]

  expression = ""
  for i in range(len(operand) - 1):
    expression += str(operand[i]) + operator[i % len(operator)] 
  expression += str(operand[-1])

  # Evaluate the expression using eval()
  output = eval(expression)
  return output

# Test case
result = method()
print(f"Expression: {expression}")
print(f"Result: {result}")