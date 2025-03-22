def method():
    numbers = [1, 2, -3, 4, -5, 6, -7, 8]  
    a = None
    b = None

    for num in numbers:
        if num < 0 and (a is None or num > a):
            a = num
        if num > 0 and (b is None or num < b):
            b = num

    return (a, b)

output = method()
print(output)  # Output: (-7, 1)