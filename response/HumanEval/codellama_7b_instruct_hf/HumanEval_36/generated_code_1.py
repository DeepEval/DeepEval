import math

def method(n):
    count = 0
    for i in range(n):
        if i % 11 == 0 or i % 13 == 0:
            str_i = str(i)
            for j in str_i:
                if j == '7':
                    count += 1
    return count

test_case = 1000
print(method(test_case))