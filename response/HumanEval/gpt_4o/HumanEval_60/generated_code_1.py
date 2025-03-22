def sum_to_n(n):
    # Calculate the sum of numbers from 1 to n
    return sum(range(1, n + 1))

def method():
    # Example usage of sum_to_n
    output = sum_to_n(5)  # This should return the sum of 1 + 2 + 3 + 4 + 5 = 15
    return output

# Test case
def test_sum_to_n():
    assert sum_to_n(5) == 15, "Test case failed for n = 5"
    assert sum_to_n(1) == 1, "Test case failed for n = 1"
    assert sum_to_n(0) == 0, "Test case failed for n = 0"
    assert sum_to_n(10) == 55, "Test case failed for n = 10"
    print("All test cases passed!")

# Run test case
test_sum_to_n()

# Run method to see the output
print(method())