def method(n):
    # Number of cars moving left to right is n
    # Number of cars moving right to left is also n
    # Every car from the left will collide with every car from the right
    output = n * n  # Total collisions
    return output

# Test case for validation
if __name__ == "__main__":
    n = 5  # Example: 5 cars moving left to right and 5 cars moving right to left
    result = method(n)
    print("Number of collisions:", result)  # Expected output: 25