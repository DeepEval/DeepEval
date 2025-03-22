from typing import List


def method(l: List[int], t: int) -> bool:
    """
    Return True if all numbers in the list l are below threshold t.
    """
    for num in l:
        if num >= t:
            return False

    return True


# Test case:
l = [1, 2, 3, 4, 5]
t = 4

output = method(l, t)

print(output)