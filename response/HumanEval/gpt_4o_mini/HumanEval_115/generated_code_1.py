def method(wells_grid, bucket_capacity):
    total_water = 0
    
    # Count the total water in the wells
    for row in wells_grid:
        total_water += sum(row)  # Count the number of 1s in the row (units of water)
    
    # Calculate the number of times we need to lower the buckets
    if bucket_capacity <= 0:
        raise ValueError("Bucket capacity must be greater than 0")
    
    # Lowering the buckets needed
    lowering_times = (total_water + bucket_capacity - 1) // bucket_capacity  # This is a ceiling division
    
    return lowering_times

# Test case for validation
wells_grid = [
    [1, 0, 1, 1],  # 3 units of water
    [0, 1, 0, 0],  # 1 unit of water
    [1, 1, 1, 1]   # 4 units of water
]
bucket_capacity = 2  # Each bucket can hold 2 units of water

output = method(wells_grid, bucket_capacity)
print(output)  # Expected output: 4 (3 from the first row, 1 from the second, and 4 from the third)