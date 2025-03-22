import itertools

def triples_sum_to_zero(nums):
    """
    triples_sum_to_zero takes a list of integers as an input. it returns True if there are three distinct elements in the list that sum to zero, and False otherwise.
    """
    for i in itertools.combinations(nums, 3):
        if sum(i) == 0:
            return True
    return False


def method():
    nums = [-1, 0, 1, 2, -3]
    output = triples_sum_to_zero(nums)
    return output


print(method())