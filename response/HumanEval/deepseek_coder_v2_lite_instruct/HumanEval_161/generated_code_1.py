def method():
    s = "aBcDeF"  # Example input
    result = []
    has_letters = False

    for char in s:
        if char.isalpha():
            has_letters = True
            if char.islower():
                result.append(char.upper())
            else:
                result.append(char.lower())
        else:
            result.append(char)

    if not has_letters:
        result.reverse()

    output = ''.join(result)
    return output

# Test case
print(method())  # Expected output: "FdCbA"