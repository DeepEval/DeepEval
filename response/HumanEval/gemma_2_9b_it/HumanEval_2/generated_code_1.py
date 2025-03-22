def method():
    # num = float(input("Enter a positive floating point number: "))

    # 修改为固定值
    num = 3.14159
    
    integer_part = int(num)
    decimal_part = num - integer_part
    return decimal_part

output = method()
print("The decimal part of the number is:", output) 

# Test Case
# Input: 3.14159
# Output: 0.14159