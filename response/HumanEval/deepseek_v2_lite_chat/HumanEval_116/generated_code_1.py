def method(arr):
    # Sort the array first by the number of '1's in the binary representation,
    # then by the actual value of the number.
    return sorted(arr, key=lambda x: (bin(x).count('1'), x))