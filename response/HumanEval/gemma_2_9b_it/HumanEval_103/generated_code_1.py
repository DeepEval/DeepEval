def method():
    # n = int(input("Enter the first integer (n): "))
    # m = int(input("Enter the second integer (m): "))
    
    # 修改为固定值
    n = 1
    m = 10

    if n > m:
        return -1

    sum_of_integers = sum(range(n, m + 1))
    average = sum_of_integers / (m - n + 1)
    rounded_average = round(average)
    binary_average = bin(rounded_average)

    return binary_average

# Test case
output = method()
print(f"The binary representation of the average is: {output}")