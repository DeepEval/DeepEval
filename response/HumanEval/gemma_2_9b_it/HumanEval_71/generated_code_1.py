import math

def method(a, b, c):
    if (a + b > c) and (a + c > b) and (b + c > a):
      # Calculate the semi-perimeter
      s = (a + b + c) / 2

      # Calculate the area using Heron's formula
      area = math.sqrt(s * (s - a) * (s - b) * (s - c))
      return round(area, 2)
    else:
      return -1

# Test case
a = 3
b = 4
c = 5
output = method(a, b, c)
print(f"The area of the triangle with sides {a}, {b}, and {c} is: {output}")