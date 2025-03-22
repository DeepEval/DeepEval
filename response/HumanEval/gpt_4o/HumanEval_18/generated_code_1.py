def method(original_string, substring):
    count = 0
    start = 0

    while start <= len(original_string) - len(substring):
        pos = original_string.find(substring, start)
        if pos != -1:
            count += 1
            start = pos + 1
        else:
            break

    return count

# Test case for validation
original_string = "abababab"
substring = "aba"
output = method(original_string, substring)
print(output)  # Expected output: 3