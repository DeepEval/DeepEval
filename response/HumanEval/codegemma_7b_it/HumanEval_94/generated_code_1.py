from sympy import isprime

def method():
    lst = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    max_prime = max(filter(isprime, lst))
    return sum(map(int, str(max_prime)))

output = method()
print(output)