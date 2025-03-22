def method():
    # a = int(input("Enter the first integer: "))
    # b = int(input("Enter the second integer: "))

    # 修改为固定值
    a = 12
    b = 18

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    output = gcd(a, b)
    return output

# Test case
if __name__ == "__main__":
    result = method()
    print(f"The greatest common divisor of the two integers is: {result}")