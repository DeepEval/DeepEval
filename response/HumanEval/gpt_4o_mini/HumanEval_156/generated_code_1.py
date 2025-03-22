def method(num):
    # Mapping of integers to their corresponding Roman numerals
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syms = [
        "m", "cm", "d", "cd",
        "c", "xc", "l", "xl",
        "x", "ix", "v", "iv",
        "i"
    ]
    
    # Initialize the Roman numeral string
    roman_numeral = ""
    
    # Iterate through the values and symbols
    for i in range(len(val)):
        # While num is greater than or equal to the value
        while num >= val[i]:
            # Append the symbol
            roman_numeral += syms[i]
            # Decrease num by the value
            num -= val[i]
    
    return roman_numeral

# Test case for validation
test_num = 1987
output = method(test_num)
print(output)  # Expected output: "mcmlxxxvii"