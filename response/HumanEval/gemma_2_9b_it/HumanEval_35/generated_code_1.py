def method():
  """
  Returns the maximum element in a list.
  """
  numbers = [1, 5, 2, 8, 3, 9, 4, 7, 6]
  maximum = numbers[0]  

  for number in numbers:
    if number > maximum:
      maximum = number

  return maximum 

output = method()
print(f"The maximum element in the list is: {output}")