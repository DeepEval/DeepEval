def method(total_carrots, eaten_carrots, required_carrots):
    """
    Calculate the total number of eaten carrots and the number of carrots left after meals.

    Args:
    total_carrots (int): The total number of carrots.
    eaten_carrots (int): The number of carrots already eaten.
    required_carrots (int): The number of carrots needed to complete the day's meals.

    Returns:
    list: A list containing the total number of eaten carrots and the number of carrots left after meals.
    """

    # Calculate the total number of carrots needed
    total_needed = eaten_carrots + required_carrots
    
    # Check if there are enough remaining carrots
    if total_needed <= total_carrots:
        # If enough, calculate the total number of eaten carrots and the number of carrots left
        eaten_total = total_needed
        carrots_left = total_carrots - total_needed
    else:
        # If not enough, eat all remaining carrots and note that you will still be hungry
        eaten_total = total_carrots
        carrots_left = 0

    # Return the result as a list
    return [eaten_total, carrots_left]

# Test case
total_carrots = 20
eaten_carrots = 10
required_carrots = 5
output = method(total_carrots, eaten_carrots, required_carrots)
print(output)  # Output: [15, 5]