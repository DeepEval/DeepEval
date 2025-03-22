def method():
    # x = int(input("Enter an integer: "))
    # shift = int(input("Enter the number of digits to shift: "))

    # 修改为固定值
    x = 123456
    shift = 2
    
    num_digits = len(str(x))
    
    if shift > num_digits:
        return str(x)[::-1]  # Reverse the digits if shift is greater than the number of digits

    shifted_str = str(x)[shift:] + str(x)[:shift] 
    return shifted_str

output = method()
print(f"Circular shifted digits: {output}")