def method():
    # s = input("Enter a string: ")
    
    # 修改为固定值
    s = "hElLo WoRlD"

    if len(s) < 3:
        return "Not happy, length is less than 3"
    
    for i in range(len(s) - 2):
        if s[i] == s[i + 1] == s[i + 2]:
            return "Not happy, not all 3 consecutive letters are distinct"

    return "Happy"


# Test case
output = method()
print(output)