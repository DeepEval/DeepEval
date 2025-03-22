def method():
    # string = input("Enter a string: ")

    # 修改为固定值
    string = "Racecar"
    
    string = string.lower().replace(" ", "")
    return string == string[::-1]

output = method()

if output:
    print("The string is a palindrome.")
else:
    print("The string is not a palindrome.")


# Test Case
# Enter a string: Racecar
# The string is a palindrome.

# Enter a string: hello world
# The string is not a palindrome.