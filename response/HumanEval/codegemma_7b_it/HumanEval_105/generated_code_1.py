import re

def method():
  # Sample input array
  input_array = [4, 9, 2, 7, 5, 3, 1, 8]

  # Sort the integers between 1 and 9 inclusive
  sorted_array = [i for i in input_array if 1 <= i <= 9]
  sorted_array.sort()

  # Reverse the sorted array
  reversed_array = sorted_array[::-1]

  # Replace each digit by its corresponding name
  output = []
  for digit in reversed_array:
    if digit == 1:
      output.append("One")
    elif digit == 2:
      output.append("Two")
    elif digit == 3:
      output.append("Three")
    elif digit == 4:
      output.append("Four")
    elif digit == 5:
      output.append("Five")
    elif digit == 6:
      output.append("Six")
    elif digit == 7:
      output.append("Seven")
    elif digit == 8:
      output.append("Eight")
    elif digit == 9:
      output.append("Nine")

  return output

# Test case
input_array = [4, 9, 2, 7, 5, 3, 1, 8]
output = method()

# Print the output
print(output)