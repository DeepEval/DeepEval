def method(n):
    product = 1
    has_odd = False
    while n > 0:
        digit = n % 10
        if digit % 2 != 0:
            product *= digit
            has_odd = True
        n //= 10
    if has_odd:
        return product
    else:
        return 0

# Test case
n = 12345
output = method(n)
print(f"The product of odd digits in {n} is: {output}")