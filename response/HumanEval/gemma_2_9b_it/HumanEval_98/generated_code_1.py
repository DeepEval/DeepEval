def method():
    # s = input("Enter a string: ")

    # 修改为固定值
    s = "hElLo WoRlD"
    
    count = 0
    for i in range(len(s)):
        if i % 2 == 0 and s[i] in "AEIOU":
            count += 1
    return count

output = method()
print("Number of uppercase vowels at even indices:", output)

# Test Case
test_string = "hElLo WoRlD"
test_output = method()
print(f"Test Case: {test_string} - Output: {test_output}")