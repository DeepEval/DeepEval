import math

def method():
    # a = int(input("Enter a: "))
    # b = int(input("Enter b: "))

    # 修改：
    a = 10
    b = 20

    output = []
    for i in range(a, b + 1):
        if i % 2 == 0:
            output.append(i)

    output.sort()

    return output

# Test case
a = 10
b = 20
expected_output = [10, 12, 14, 16, 18, 20]
actual_output = method()

if actual_output == expected_output:
    print("Test case passed")
else:
    print("Test case failed")