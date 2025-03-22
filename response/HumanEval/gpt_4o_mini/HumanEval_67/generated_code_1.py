def method(fruit_string, total_fruits):
    # Split the input string to get apples and oranges
    fruit_counts = fruit_string.split(',')
    
    # Convert the counts of apples and oranges to integers
    apples = int(fruit_counts[0].strip())
    oranges = int(fruit_counts[1].strip())
    
    # Calculate the number of mangoes
    mangoes = total_fruits - (apples + oranges)
    
    return mangoes

# Test case
fruit_string = "5, 3"  # 5 apples and 3 oranges
total_fruits = 15  # Total fruits in the basket
output = method(fruit_string, total_fruits)
print(f"Number of mangoes in the basket: {output}")