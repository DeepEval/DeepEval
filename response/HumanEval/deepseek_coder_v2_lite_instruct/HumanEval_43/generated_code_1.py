def pairs_sum_to_zero(lst):
    seen = set()
    for num in lst:
        if -num in seen:
            return True
        seen.add(num)
    return False

def method():
    # Test case
    test_list = [1, 2, -2, 3, -3]
    output = pairs_sum_to_zero(test_list)
    return output

# Run the method to get the output
print(method())  # Expected output: True