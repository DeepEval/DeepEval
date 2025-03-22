import math

def method():
    def is_prime(num):
        if num <= 1:
            return False
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                return False
        return True

    def fibonacci(n):
        a = 0
        b = 1
        for i in range(n):
            a, b = b, a + b
        return a

    # n = int(input("Enter the value of n: "))

    # 修改为固定值
    n = 5
    
    i = 0
    while True:
        fib_num = fibonacci(i)
        if is_prime(fib_num):
            if i == n - 1:
                output = fib_num
                return output
        i += 1

    
# Test Case
n = 5
output = method()
print(f"The {n}-th Fibonacci number that is also prime is: {output}")