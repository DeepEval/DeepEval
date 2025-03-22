def method(numbers):
    """
    Given a list of numbers, return the sum of squares of the numbers in the list that are odd. 
    Ignore numbers that are negative or not integers.
    
    For example:
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0
    If the input list is empty, return 0.
    """
    sum_of_squares = 0
    for number in numbers:
        if isinstance(number, int) and number > 0 and number % 2 != 0:
            sum_of_squares += number * number
    return sum_of_squares

# Test case
print(method([1, 3, 2, 0]))  # Output: 10
print(method([-1, -2, 0]))   # Output: 0
print(method([9, -2]))      # Output: 81
print(method([0]))          # Output: 0