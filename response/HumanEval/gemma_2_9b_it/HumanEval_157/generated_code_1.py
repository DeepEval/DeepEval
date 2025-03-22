import math

def method():
    # a = float(input("Enter the length of side a: "))
    # b = float(input("Enter the length of side b: "))
    # c = float(input("Enter the length of side c: "))

    # 修改为固定值
    a = 3
    b = 4
    c = 5

    # Check if the triangle inequality holds
    if a + b <= c or a + c <= b or b + c <= a:
        return False

    # Calculate the square of the hypotenuse
    c_squared = c * c

    # Check if the Pythagorean theorem holds
    return a * a + b * b == c_squared

output = method()
print("Is it a right-angled triangle?", output)