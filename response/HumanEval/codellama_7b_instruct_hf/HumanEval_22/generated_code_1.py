# Import necessary packages
import re

def method():
    # Given list of values
    values = ["apple", 123, "banana", 456, "cherry", 789]

    # Filter values for integers using regular expressions
    filtered_values = [value for value in values if isinstance(value, int)]

    # Return filtered values
    return filtered_values

# Test case for validation
test_values = ["apple", 123, "banana", 456, "cherry", 789]
filtered_test_values = method()
# assert filtered_test_values == [123, 456, 789]