from math import factorial

def method():

    # n = int(input("Enter the size of the list: "))
    # 修改：
    n = 5
    
    output = []

    for i in range(1, n + 1):
        if i % 2 == 0:
            output.append(factorial(i))
        else:
            sum_of_numbers = 0
            for j in range(1, i + 1):
                sum_of_numbers += j
            output.append(sum_of_numbers)

    return output


# Test case
input_size = 5
expected_output = [120, 6, 24, 10, 15]

output = method()

if output == expected_output:
    print("Test case passed.")
else:
    print("Test case failed.")