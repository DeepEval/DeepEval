import math

def method(n):
    output = []
    level = n
    for i in range(n):
        output.append(level)
        if (level % 2 == 0):
            level = level + 1
        else:
            level = level + 2

    return output

# Test Case
n = 5
print(method(n))