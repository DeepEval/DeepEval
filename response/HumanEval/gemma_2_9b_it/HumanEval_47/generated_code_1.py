import statistics

def method(l):
  """
  Returns the median of elements in the list l.
  """
  output = statistics.median(l)
  return output

# Test case
test_list = [1, 3, 2, 4, 5]
median = method(test_list)
print(f"The median of {test_list} is: {median}")