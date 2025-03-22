import math

def method(lst):

    output = 0

    for i in range(0, len(lst)):
        if (i % 2 == 0 and lst[i] % 2 == 0):
            output += lst[i]

    return output


def test_method():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # assert method(lst) == 20
    print(method(lst))


test_method()