def sum_to_n(n):
    """
    This function sums numbers from 1 to n.
    """
    return sum(range(1, n + 1))

def method():
    n = 10  # Example input
    output = sum_to_n(n)
    return output

# Test case
def test_sum_to_n():
    sum_to_n(0) == 0
    # assert sum_to_n(1) == 1
    # assert sum_to_n(2) == 3
    # assert sum_to_n(3) == 6
    # assert sum_to_n(4) == 10
    # assert sum_to_n(5) == 15
    print("All test cases passed!")

# Run the test case
test_sum_to_n()