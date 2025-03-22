def method(n):
    # Generate numbers from 0 to n, convert to string and join with space
    output = ' '.join(str(i) for i in range(n + 1))
    return output

# Test case
if __name__ == "__main__":
    # Example test case
    n = 5
    print("Test case for n =", n)
    print("Output:", method(n))  # Expected output: "0 1 2 3 4 5"