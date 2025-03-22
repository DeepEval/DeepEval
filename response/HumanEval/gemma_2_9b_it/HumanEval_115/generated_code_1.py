def method():
    wells = [
        [1, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 1]
    ]
    bucket_capacity = 1  
    num_buckets = len(wells)

    total_trips = 0
    for row in wells:
      total_trips += sum(row) // bucket_capacity 
      if sum(row) % bucket_capacity:
        total_trips += 1  
    return total_trips

# Test Case
output = method()
print(f"Number of bucket trips required: {output}")