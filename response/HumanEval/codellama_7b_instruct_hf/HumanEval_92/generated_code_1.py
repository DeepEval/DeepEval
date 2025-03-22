import math

def method():
    # Get the three numbers from the user
    # num1 = int(input("Enter the first number: "))
    # num2 = int(input("Enter the second number: "))
    # num3 = int(input("Enter the third number: "))

    # 修改
    num1 = 3
    num2 = 4
    num3 = 5

    # Check if one of the numbers is equal to the sum of the other two
    if num1 == num2 + num3 or num2 == num1 + num3 or num3 == num1 + num2:
        return True
    else:
        return False

# Test case
print(method())