def method(str):
    str = str.lower()
    str = str.replace(" ", "")
    return str == str[::-1]


test_case = "madam"
output = method(test_case)
print(output)