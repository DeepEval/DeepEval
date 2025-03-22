def method():
    # num = int(input("Enter a positive integer (1-1000): "))

    # 修改为固定值
    num = 1234
    
    roman_values = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    int_values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]

    roman_numeral = ''
    i = 0
    while num > 0:
        for _ in range(num // int_values[i]):
            roman_numeral += roman_values[i]
            num -= int_values[i]
        i += 1
    
    output = roman_numeral.lower()
    return output

# Test Case
print(method())