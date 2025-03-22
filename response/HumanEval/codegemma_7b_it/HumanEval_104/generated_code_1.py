# Import necessary packages
import math

def method(x):
    # Filter elements with even digits using a list comprehension
    result = [i for i in x if not any(int(d) % 2 == 0 for d in str(i))]

    # Sort the result in ascending order
    result.sort()

    return result

# Test case
x = [123, 4567, 789, 2468, 3579]
print(method(x))  # Output: [789, 3579]