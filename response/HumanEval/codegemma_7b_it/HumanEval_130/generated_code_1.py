import math

def method():
    # n = int(input("Enter the nth term of the Tribonacci sequence: "))
    # 修改：
    n =10
    tri = [0] * (n + 1)
    tri[1] = 3
    tri[2] = 1.5

    for i in range(3, n + 1):
        if i % 2 == 0:
            tri[i] = 1 + i / 2
        else:
            tri[i] = tri[i - 1] + tri[i - 2] + tri[i + 1]

    return tri[n]

def test_case():
    test_case_input = 10
    test_case_output = 448.5
    method_output = method()
    # assert method(test_case_input) == test_case_output, "Test case failed"


test_case()