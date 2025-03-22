def method():
    # Define Roman numeral mappings
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    
    # Function to convert integer to Roman numeral
    def int_to_roman(num):
        roman_num = ''
        i = 0
        while num > 0:
            for _ in range(num // val[i]):
                roman_num += syms[i]
                num -= val[i]
            i += 1
        return roman_num
    
    # Get user input
    # num = int(input("Enter a positive integer between 1 and 1000: "))
    num = 900
    
    # Check if the input is within the valid range
    if 1 <= num <= 1000:
        # Convert to Roman numeral and return in lowercase
        output = int_to_roman(num).lower()
        return output
    else:
        return "Number out of range. Please enter a number between 1 and 1000."

# Test case
print(method())