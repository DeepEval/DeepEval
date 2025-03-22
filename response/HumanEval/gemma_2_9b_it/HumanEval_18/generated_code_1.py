def method():
    # string = input("Enter the original string: ")
    # substring = input("Enter the substring: ")

    # 修改为固定值
    string = "Hello, World!"
    substring = "o"

    count = string.count(substring)
    return count

# Test Case
output = method()
print(output)
# print(f"The substring '{substring}' appears {output} times in the string.")