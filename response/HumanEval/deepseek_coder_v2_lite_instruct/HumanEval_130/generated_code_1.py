def method():
    # Define the Tribonacci sequence function
    def tri(n, memo={}):
        if n in memo:
            return memo[n]
        if n == 1:
            return 3
        elif n % 2 == 0:
            memo[n] = 1 + n // 2
        else:
            memo[n] = tri(n - 1, memo) + tri(n - 2, memo) + tri(n + 1, memo)
        return memo[n]

    # Example test case
    n = 5
    output = tri(n)
    return output

# Test the method function
if __name__ == "__main__":
    result = method()
    print(f"The Tribonacci sequence value for n=5 is: {result}")