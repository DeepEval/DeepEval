def method():
    def sum_to_n(n):
        total_sum = 0
        for i in range(1, n + 1):
            total_sum += i
        return total_sum
    return sum_to_n

# Example usage:
output = method()()  # Call the returned function with no argument
print(output)        # Output: The sum of numbers from 1 to n

# Test case
# assert output == (n * (n + 1)) // 2, "Test failed"