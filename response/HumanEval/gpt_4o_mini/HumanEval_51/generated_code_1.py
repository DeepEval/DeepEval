def remove_vowels(input_string):
    vowels = "aeiouAEIOU"
    return ''.join(char for char in input_string if char not in vowels)

def method():
    test_string = "Hello, World!"
    output = remove_vowels(test_string)
    return output

# Test case for validation
if __name__ == "__main__":
    result = method()
    print("Output after removing vowels:", result)  # Expected output: "Hll, Wrld!"