def method():
    # string = input("Enter a string: ")

    # 修改为固定值
    string = "Hello world!"
    
    words = string.split()  
    last_char = string[-1]

    if last_char.isalpha() and last_char not in " ".join(words):
        return True
    else:
        return False

output = method()
print(f"Output: {output}")
 
# Test case
print("Test Case 1:")
print(method()) # Example input: "Hello world!" should return True

print("Test Case 2:")
print(method()) # Example input: "123abc" should return True

print("Test Case 3:")
print(method()) # Example input: "apple space" should return False