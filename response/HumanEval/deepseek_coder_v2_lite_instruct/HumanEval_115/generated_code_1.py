def method():
    # Define the grid of wells
    grid = [
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [1, 1, 1, 0]
    ]
    
    # Initialize the count of bucket lowerings
    bucket_lowerings = 0
    
    # Iterate through each well (row) in the grid
    for row in grid:
        # Count the number of units of water in the current well
        water_units = row.count(1)
        # Add the number of water units to the total bucket lowerings
        bucket_lowerings += water_units
    
    # Return the total number of bucket lowerings
    return bucket_lowerings

# Test the method with the provided grid
output = method()
print("Number of times to lower the bucket:", output)