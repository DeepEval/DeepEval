import statistics

def method(l):
    # Calculate the median using the statistics module
    output = statistics.median(l)
    return output

# Test case for validation
if __name__ == "__main__":
    # Example list
    test_list = [3, 5, 1, 4, 2]
    # Expected median: 3
    print("Median of test_list:", method(test_list))  # Output should be 3