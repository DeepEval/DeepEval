def method():
  
  data = [3, 1, 6, 4, 2]

  def sum_of_digits(num):
    return sum(int(digit) for digit in str(num))

  output = sorted(data, key=lambda num: (sum_of_digits(num), data.index(num)))
  return output

# Test Case
print(method()) 
# Expected Output: [1, 2, 3, 4, 6]