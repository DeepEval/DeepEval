def method():
    # n = int(input("Enter a positive integer: "))

    # 修改为固定值
    n = 60
    
    factors = {}
    i = 2
    while i * i <= n:
        count = 0
        while n % i == 0:
            count += 1
            n //= i
        if count > 0:
            factors[i] = count
        i += 1
    if n > 1:
        factors[n] = 1

    output = list(factors.keys())
    
    return output

# Test Case
output = method()
print("Prime factors of the input number (in order):", output)