def method(n):
    # Base case
    if n == 1:
        return 3
    # Create a dictionary to memoize results of previous computations
    memo = {1: 3}

    def tribonacci(num):
        if num in memo:
            return memo[num]
        if num % 2 == 0:
            memo[num] = 1 + num / 2
        else:
            memo[num] = tribonacci(num - 1) + tribonacci(num - 2) + tribonacci(num + 1)
        return memo[num]
    
    output = tribonacci(n)
    return output

# Test case
print(method(1))  # Output should be 3
print(method(2))  # Output should be 2 (since 1 + 2/2 = 2)
print(method(3))  # This will compute based on the recursive rule for odd numbers