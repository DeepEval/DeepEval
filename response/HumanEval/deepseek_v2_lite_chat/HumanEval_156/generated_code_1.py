def method(num):
    # Dictionary to map integer values to Roman numeral characters
    int_to_roman = {
        1000: "M", 900: "CM", 500: "D", 400: "CD", 100: "C", 90: "XC",
        50: "L", 40: "XL", 10: "X", 9: "IX", 5: "V", 4: "IV", 1: "I"
    }
    
    # Initialize an empty string to hold the Roman numeral
    roman_numeral = ""
    
    # Iterate through the dictionary in reverse order
    for value in sorted(int_to_roman.keys(), reverse=True):
        while num >= value:
            # Append the corresponding Roman numeral character
            roman_numeral += int_to_roman[value]
            # Subtract the value from num
            num -= value
    
    return roman_numeral

# Test case
def test_method():
    # assert method(1) == "I"
    # assert method(2) == "II"
    # assert method(3) == "III"
    # assert method(4) == "IV"
    # assert method(5) == "V"
    # assert method(9) == "IX"
    # assert method(10) == "X"
    # assert method(40) == "XL"
    # assert method(50) == "L"
    # assert method(90) == "XC"
    # assert method(100) == "C"
    # assert method(400) == "CD"
    # assert method(500) == "D"
    # assert method(900) == "CM"
    # assert method(1000) == "M"
    # assert method(3999) == "MMMCMXCIX"
    print(method(3999))

test_method()