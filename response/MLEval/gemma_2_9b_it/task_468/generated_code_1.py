def method():
  """
  This function determines if a given number is divisible by 7.
  """
  # number = int(input("Enter a number: "))
  number = 14
  
  if number % 7 == 0:
    output = f"{number} is divisible by 7"
  else:
    output = f"{number} is not divisible by 7"
  
  return output

# Call the method and print the result
result = method()
print(result)