def method(N):
    # Convert the integer N to its binary representation and count the '1's
    binary_representation = bin(N)  # Converts N to binary, e.g. bin(5) returns '0b101'
    digit_sum = binary_representation.count('1')  # Count how many '1's are in the binary representation
    
    return digit_sum

# Test case for validation
if __name__ == "__main__":
    N = 5  # Binary representation is '101', which has two '1's
    output = method(N)
    print(f"The total sum of the digits in binary for {N} is: {output}")  # Expected output: 2