def method(arr):
    n = len(arr)
    changes = 0
    
    for i in range(n // 2):
        if arr[i] != arr[n - 1 - i]:
            changes += 1
    
    return changes

# Test case for validation
if __name__ == "__main__":
    test_array = [1, 2, 3, 4, 5]  # Example array
    output = method(test_array)
    print(f"Minimum number of changes required: {output}")  # Expected output: 2