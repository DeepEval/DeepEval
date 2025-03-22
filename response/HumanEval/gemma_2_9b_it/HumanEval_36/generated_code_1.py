def method(n):
    count = 0
    for i in range(1, n):
        if (i % 11 == 0 or i % 13 == 0) and '7' in str(i):
            count += 1
    return count

# Test Case
n = 100
output = method(n)
print(f"Number of times 7 appears in integers less than {n} divisible by 11 or 13: {output}")