def method(n):
    # Using a dictionary to memoize results
    memo = {}

    def fibfib(n):
        # Base cases
        if n == 0:
            return 0
        if n == 1:
            return 0
        if n == 2:
            return 1

        # Check if the value is already computed
        if n in memo:
            return memo[n]

        # Recursive case with memoization
        memo[n] = fibfib(n - 1) + fibfib(n - 2) + fibfib(n - 3)
        return memo[n]

    # Calculate the n-th FibFib number
    output = fibfib(n)
    return output

# Test case for validation
if __name__ == "__main__":
    test_n = 5  # Example input
    result = method(test_n)
    print(f"The {test_n}-th FibFib number is: {result}")