def method():
    def tribonacci(n):
        # Base cases
        if n == 0:
            return 3
        if n == 1:
            return 1
        if n == 2:
            return 1

        # Memoization dictionary to store computed values
        memo = {0: 3, 1: 1, 2: 1}

        # Recursive case
        for i in range(3, n + 1):
            if i % 2 == 0:
                memo[i] = 1 + i // 2
            else:
                memo[i] = memo[i - 1] + memo[i - 2] + memo[i + 1]

        return memo[n]

    # Example usage
    n = 5
    output = tribonacci(n)
    return output

# Test case to validate the function
# def test_method():
#     assert method() == 7

# Run the test case
# test_method()

method()