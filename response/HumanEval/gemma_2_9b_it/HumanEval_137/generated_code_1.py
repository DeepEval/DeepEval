def method():
    # num1 = float(input("Enter the first number: "))
    # num2 = input("Enter the second number: ")

    # 修改为固定值
    num1 = 1
    num2 = 2
    
    try:
      num2 = float(num2)
    except ValueError:
      pass
    
    if num1 > num2:
        output = num1
    elif num2 > num1:
        output = num2
    else:
        output = None

    return output

# Test Case
print(method())