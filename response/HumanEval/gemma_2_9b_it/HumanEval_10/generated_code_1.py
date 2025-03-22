def method():
    # string = input("Enter a string: ")
    # 修改
    string = "hello"
    
    # Remove spaces and convert to lowercase for case-insensitive comparison
    string = string.replace(" ", "").lower() 
    
    return string == string[::-1]

# Test case
output = method()
print(f"Is the string a palindrome? {output}")