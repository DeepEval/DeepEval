def method():
    # text = input("Enter a string: ")

    # 修改为固定值
    text = "Hello, World!"

    if ' ' in text:
        output = text.split()
    elif ',' in text:
        output = text.split(',')
    else:
        output = sum(1 for i in range(ord('a'), ord('z') + 1) if (i - ord('a')) % 2 != 0)
    return output

# Test case
result = method()
print(f"Result: {result}")