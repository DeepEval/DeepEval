def method(data):
    """
    Given a non-empty list of integers, return the sum of all of the odd elements that are in even positions. 
    """
    sum = 0
    for i in range(1, len(data), 2):
      if data[i] % 2 != 0:
        sum += data[i]
    return sum

# Test Case
test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
output = method(test_data)
print(f"Output for test_data {test_data} is: {output}")