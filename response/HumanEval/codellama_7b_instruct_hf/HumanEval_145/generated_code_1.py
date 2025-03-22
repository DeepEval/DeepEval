# Importing necessary packages
from functools import cmp_to_key

def digit_sum(number):
    """Calculates the sum of digits for a given number."""
    sum_of_digits = 0
    while number > 0:
        sum_of_digits += number % 10
        number //= 10
    return sum_of_digits

def method():
    """Sorts the list of integers in ascending order based on the sum of their digits."""
    input_list = [123, 234, 345, 456, 567, 678, 789]

    # Sorting the list using a custom comparison function
    input_list.sort(key=cmp_to_key(lambda x, y: (digit_sum(x), input_list.index(x)) - (digit_sum(y), input_list.index(y))))

    # Returning the sorted list
    return input_list

# Test case
if __name__ == "__main__":
    output = method()
    print(output)