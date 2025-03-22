from math import sqrt

def method():
    # n = int(input())
    # 修改：
    n = 5
    a = [i * i - i + 1 for i in range(1, n + 1)]
    output = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if (a[i] + a[j] + a[k]) % 3 == 0:
                    output += 1
    return output

# Test case
n = 5
a = [i * i - i + 1 for i in range(1, n + 1)]
print(a)
print(method())