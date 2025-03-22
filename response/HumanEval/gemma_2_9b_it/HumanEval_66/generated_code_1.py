def method():
    # input_string = input("Enter a string: ")

    # 修改为固定值
    input_string = "Hello, World!"
    
    sum_of_upper_ascii = 0
    for char in input_string:
        if char.isupper():
            sum_of_upper_ascii += ord(char)
    return sum_of_upper_ascii

# Test case
output = method()
print(f"Sum of ASCII codes of upper characters: {output}")