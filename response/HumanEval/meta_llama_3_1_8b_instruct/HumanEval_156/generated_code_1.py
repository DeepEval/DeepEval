def int_to_roman(num: int) -> str:
    """
    Converts an integer to its roman numeral equivalent as a string in lowercase.
    
    Args:
    num (int): A positive integer between 1 and 1000.
    
    Returns:
    str: The roman numeral equivalent of the input integer as a string in lowercase.
    """
    if not 1 <= num <= 1000:
        raise ValueError("Input must be a positive integer between 1 and 1000.")

    roman_numerals = {
        1000:'m',
        900: 'cm',
        500: 'd',
        400: 'cd',
        100: 'c',
        90: 'xc',
        50: 'l',
        40: 'xl',
        10: 'x',
        9: 'ix',
        5: 'v',
        4: 'iv',
        1: 'i'
    }

    output = ''
    for value, numeral in roman_numerals.items():
        while num >= value:
            output += numeral
            num -= value

    return output.lower()


def method():
    num = 123
    output = int_to_roman(num)
    print(f"The roman numeral equivalent of {num} is {output}.")

    return output


method()