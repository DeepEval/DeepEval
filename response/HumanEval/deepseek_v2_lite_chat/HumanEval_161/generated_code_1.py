def method():
    def reverse_case_or_reverse(s):
        # Check if the string contains any letters, if not, reverse the string
        if not any(c.isalpha() for c in s):
            # Reverse the string
            return s[::-1]
        else:
            # Reverse the case of each letter if present, otherwise keep it as is
            return ''.join([char.lower() if char.isupper() else char.upper() if char.islower() else char for char in s])

    return reverse_case_or_reverse(s)

# Example test case
s = "Hello, World!"
output = method()
print(output)  # Expected output: "hELLO, wORLD!"