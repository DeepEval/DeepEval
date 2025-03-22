def method():
    # input_string = input("Enter a string: ")

    # 修改为固定值
    input_string = "Hello WORLD!"
    
    output = ''.join([char.upper() if char.islower() else char.lower() for char in input_string])
    return output

# Test case
test_string = "Hello WORLD!"
result = method()
print(f"Input: {test_string}")
print(f"Output: {result}")