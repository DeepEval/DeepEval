def method():
    def largest_index_not_greater_or_equal(arr):
        # Iterate over the array and check for the required condition
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                return i
        
        # If no such element is found, return -1
        return -1

    # Example usage
    example_array = [3, 2, 1, 5, 6]
    output = largest_index_not_greater_or_equal(example_array)
    return output

# Example test case
print(method())