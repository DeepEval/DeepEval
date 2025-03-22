def method(num):
    # Define the mapping of integers to Roman numerals
    value_to_roman = [
        (1000, 'm'), (900, 'cm'), (500, 'd'), (400, 'cd'),
        (100, 'c'), (90, 'xc'), (50, 'l'), (40, 'xl'),
        (10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iv'),
        (1, 'i')
    ]
    
    # Initialize the result string
    roman_numeral = ''
    
    # Iterate over the mappings
    for value, roman in value_to_roman:
        while num >= value:
            roman_numeral += roman
            num -= value
    
    return roman_numeral

# Test case for validation
def test():
    # Example: Convert 1987 to Roman numeral
    num = 1987
    print(method(num))  # Expected output: 'mcmlxxxvii'

# Execute the test function
test()