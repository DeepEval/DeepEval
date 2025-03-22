def method(n):
    # Generate the list of numbers from 0 to n
    numbers = list(range(n + 1))
    # Join the numbers into a space-delimited string
    output = ' '.join(map(str, numbers))
    return output

# Test case
n = 10
print(method(n))  # Expected output: "0 1 2 3 4 5 6 7 8 9 10"