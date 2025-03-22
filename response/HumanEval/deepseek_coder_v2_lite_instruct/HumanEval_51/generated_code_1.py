def remove_vowels(s):
    vowels = "aeiouAEIOU"
    return "".join([char for char in s if char not in vowels])

def method():
    input_string = "Hello, World!"
    output = remove_vowels(input_string)
    return output

# # Test case
# def test_remove_vowels():
#     assert remove_vowels("Hello, World!") == "Hll, Wrld!"
#     assert remove_vowels("Python") == "Pythn"
#     assert remove_vowels("AEIOUaeiou") == ""
#     assert remove_vowels("12345") == "12345"
#     assert remove_vowels("Python!") == "Pythn!"
#     print("All test cases passed!")

# # Running the test case
# test_remove_vowels()

method()