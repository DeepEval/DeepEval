def method():
    """
    Calculate the number of times buckets need to be lowered to empty wells.

    The function takes a 2D list representing the wells as input. Each 1 in the list represents a unit of water.
    It returns the number of times the buckets need to be lowered to empty all wells.

    :return: int
    """

    # Define the function to calculate the number of times buckets need to be lowered
    def calculate_buckets(wells, bucket_capacity):
        # Initialize the total buckets and the current bucket
        total_buckets = 0
        current_bucket = 0

        # Iterate over each well
        for well in wells:
            # Iterate over each unit of water in the well
            for water in well:
                # If the water is 1 (i.e., a unit of water)
                if water == 1:
                    # Add the water to the current bucket
                    current_bucket += 1

                    # If the current bucket is full
                    if current_bucket == bucket_capacity:
                        # Increment the total buckets and reset the current bucket
                        total_buckets += 1
                        current_bucket = 0

        # Return the total buckets
        return total_buckets + 1  # Add 1 because we need to lower the bucket after emptying the last well

    # Define the input wells
    wells = [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    # Define the bucket capacity
    bucket_capacity = 3  # Assuming the bucket capacity is 3 units of water

    # Calculate and return the total buckets needed
    return calculate_buckets(wells, bucket_capacity)


# Test the function
print(method())