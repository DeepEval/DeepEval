import math

def method():

    # n, m = map(int, input().split())

    n = 10
    m = 20
    
    if n > m:
        return -1

    sum = 0
    count = 0

    for i in range(n, m + 1):
        sum += i
        count += 1

    average = sum / count
    rounded_average = math.ceil(average)

    binary_average = bin(rounded_average)[2:]

    return binary_average


# Test case
n = 10
m = 20

output = method()

print(output)