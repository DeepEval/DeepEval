import math

def method(grid, bucket_capacity):
    # Calculate the total number of water units in the grid
    total_water_units = sum(sum(row) for row in grid)
    
    # Calculate the number of times to lower the bucket
    # Use math.ceil to round up to the nearest whole number
    times_to_lower_bucket = math.ceil(total_water_units / bucket_capacity)
    
    return times_to_lower_bucket

# Test case
grid = [
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 0]
]
bucket_capacity = 3

# Explanation:
# Total water units = 7
# Bucket capacity = 3
# Times to lower bucket = ceil(7 / 3) = 3

output = method(grid, bucket_capacity)
print(output)  # Expected output is 3