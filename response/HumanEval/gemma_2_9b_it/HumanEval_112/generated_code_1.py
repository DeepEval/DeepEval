def method():
    # s = input("Enter the first string: ")
    # c = input("Enter the second string: ")

    # 修改为固定值
    s = "A"
    c = "a"

    result = ''.join([char for char in s if char not in c])
    
    is_palindrome = result == result[::-1] 

    output = (result, is_palindrome)
    return output

# Test case
output = method()
print(f"Result string: {output[0]}")
print(f"Is palindrome: {output[1]}")