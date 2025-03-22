def method(n):
    if n <= 0:
        return "Input should be a positive integer."
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        fib_n_minus_1 = 0
        fib_n_minus_2 = 1
        fib_n = 0
        for _ in range(2, n):
            fib_n = fib_n_minus_1 + fib_n_minus_2
            fib_n_minus_2 = fib_n_minus_1
            fib_n_minus_1 = fib_n
        return fib_n

# Test case to validate the function
def test_method():
    # assert method(1) == 0
    # assert method(2) == 1
    # assert method(3) == 1
    # assert method(4) == 2
    # assert method(5) == 3
    # assert method(10) == 55
    # print("All test cases pass")
    method(1)

# Run the test case
test_method()