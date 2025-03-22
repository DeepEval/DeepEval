def method(eaten_carrots, total_carrots, carrots_per_meal):
    """
    Simulates a rabbit eating carrots.

    Args:
      eaten_carrots: The number of carrots already eaten.
      total_carrots: The total number of carrots available.
      carrots_per_meal: The number of carrots needed for a meal.

    Returns:
      A list containing [total eaten carrots, remaining carrots]. 
    """

    remaining_carrots = total_carrots - eaten_carrots
    meals_eaten = remaining_carrots // carrots_per_meal
    
    total_eaten = eaten_carrots + meals_eaten * carrots_per_meal
    remaining = remaining_carrots % carrots_per_meal 

    return [total_eaten, remaining]

# Test case
eaten_carrots = 5
total_carrots = 15
carrots_per_meal = 3

output = method(eaten_carrots, total_carrots, carrots_per_meal)
print(f"Eaten: {output[0]}, Remaining: {output[1]}")