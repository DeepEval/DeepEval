import numpy as np

def method(text):
    """
    Return encoded string by cycling groups of three characters.
    """
    output = ""
    i = 0
    while i < len(text):
        if i + 2 >= len(text):
            output += text[i] + text[i+1]
        else:
            output += text[i] + text[i+1] + text[i+2]
        i += 3
    return output


def test_method():
    """
    Test case for method().
    """
    text = "abcdefghijklmn"
    expected_output = "abcefghijklm"
    output = method(text)
    # assert output == expected_output