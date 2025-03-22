def method():
    # a = input("Enter the first string (consisting of 1s and 0s): ")
    # b = input("Enter the second string (consisting of 1s and 0s): ")
    
    # 修改为固定值
    a = "10101"
    b = "11011"

    output = bin(int(a, 2) ^ int(b, 2))[2:] 
    return output

# Test Case
result = method()
print(f"XOR of the input strings is: {result}")