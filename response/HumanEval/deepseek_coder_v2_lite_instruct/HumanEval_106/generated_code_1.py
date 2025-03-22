def method():
    def factorial(n):
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def sum_up_to(n):
        return n * (n + 1) // 2

    n = 10  # Example parameter value
    output = []
    for i in range(1, n + 1):
        if i % 2 == 0:
            output.append(factorial(i))
        else:
            output.append(sum_up_to(i))
    return output

# Test the function with a specific value of n
test_output = method()
print(test_output)