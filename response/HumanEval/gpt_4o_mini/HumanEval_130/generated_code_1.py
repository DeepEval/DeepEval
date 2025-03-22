def method(n):
    # Define a dictionary to store computed values for memoization
    memo = {}

    def tri(n):
        # Base case for n = 1
        if n == 1:
            return 3
        # Check if the value is already computed
        if n in memo:
            return memo[n]
        
        # Calculate the Tribonacci value based on whether n is even or odd
        if n % 2 == 0:  # n is even
            result = 1 + n / 2
        else:  # n is odd
            result = tri(n - 1) + tri(n - 2) + tri(n + 1)
        
        # Store the computed value in the memoization dictionary
        memo[n] = result
        return result

    # Calculate the Tribonacci value for the given n
    output = tri(n)
    return output

# Test case
if __name__ == "__main__":
    test_value = 5
    result = method(test_value)
    print(f'The Tribonacci value for n={test_value} is {result}')