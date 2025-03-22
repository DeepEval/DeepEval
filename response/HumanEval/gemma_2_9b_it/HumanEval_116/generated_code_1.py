def method():
    def count_ones(n):
        return bin(n).count("1")

    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    # Sort based on count of ones, then decimal value
    numbers.sort(key=lambda x: (count_ones(x), x))

    return numbers 

# Test case
output = method()
print(f"Sorted array: {output}")  # Expected output: [0, 1, 2, 4, 8, 3, 5, 6, 7]