def method():
    # decimal_num = int(input("Enter a decimal number: "))
    
    # 修改为固定值
    decimal_num = 10

    binary_num = bin(decimal_num)[2:]  # Convert to binary, remove "0b" prefix
    output = "db" + binary_num + "db"
    return output

# Test case
result = method()
print(f"Binary representation: {result}")