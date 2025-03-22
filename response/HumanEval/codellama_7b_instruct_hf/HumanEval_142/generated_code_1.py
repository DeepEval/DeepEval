from functools import reduce

def method(int_list):
    """
    This function takes a list of integers and squares the integer entry if its index is a multiple of 3 and cubes the integer entry if its index is a multiple of 4 and not a multiple of 3. The function then returns the sum of all entries.

    Args:
        int_list: A list of integers.

    Returns:
        The sum of all entries in the list.
    """

    def square_if_multiple_of_3(x, i):
        if i % 3 == 0:
            return x ** 2
        else:
            return x

    def cube_if_multiple_of_4_not_multiple_of_3(x, i):
        if i % 4 == 0 and i % 3 != 0:
            return x ** 3
        else:
            return x

    output = reduce(lambda x, y: x + y, map(square_if_multiple_of_3, int_list, range(len(int_list))))
    output = reduce(lambda x, y: x + y, map(cube_if_multiple_of_4_not_multiple_of_3, int_list, range(len(int_list))))

    return output


test_case = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(method(test_case))