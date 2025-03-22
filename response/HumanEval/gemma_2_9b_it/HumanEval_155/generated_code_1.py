def method():
    # num = int(input("Enter an integer: "))
    
    # 修改为固定值
    num = 123456789

    even_count = 0
    odd_count = 0
    for digit in str(num):
        if int(digit) % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    output = (even_count, odd_count)
    return output

# Test Case
result = method()
print(f"Number of even digits: {result[0]}")
print(f"Number of odd digits: {result[1]}")