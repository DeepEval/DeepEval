def method():
    n = 100  # Example test case
    output = 0

    for i in range(1, n):
        if i % 11 == 0 or i % 13 == 0:
            output += str(i).count('7')

    return output

# Test case
print(method())  # Expected output for n = 100