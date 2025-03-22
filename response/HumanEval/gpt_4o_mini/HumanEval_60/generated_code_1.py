def sum_to_n(n):
    """Function to sum numbers from 1 to n."""
    return sum(range(1, n + 1))

def method():
    n = 10  # You can change this value to test with different inputs
    output = sum_to_n(n)
    return output

# Test case
if __name__ == "__main__":
    result = method()
    print(f"The sum of numbers from 1 to 10 is: {result}")  # Expected output: 55