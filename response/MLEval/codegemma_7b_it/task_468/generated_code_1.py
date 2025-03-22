import numpy as np

def method():

    # Get the input number from the user
    # number = int(input("Enter a number: "))

    # 修改为固定值
    number = 14

    # Check if the number is divisible by 7
    if number % 7 == 0:
        output = "The number is divisible by 7."
    else:
        output = "The number is not divisible by 7."

    return output

# Call the method
print(method())