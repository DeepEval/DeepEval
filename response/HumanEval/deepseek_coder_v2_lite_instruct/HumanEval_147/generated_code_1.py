def method():
    # n = int(input())  # Read the value of n from input
    n = 5  # Hardcode the value of n
    a = [i * i - i + 1 for i in range(1, n + 1)]  # Generate the array a
    
    count = 0  # Initialize the count of valid triples
    
    # Iterate through all possible triples (i, j, k)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (a[i] + a[j] + a[k]) % 3 == 0:
                    count += 1
    
    output = count  # The final output is the count of valid triples
    return output

# Test case
print(method())  # Expected output for n = 5 is 2