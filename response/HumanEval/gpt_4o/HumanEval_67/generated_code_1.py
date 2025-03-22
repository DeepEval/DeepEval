def method(fruit_string, total_fruits):
    # Parse the string to get the number of apples and oranges
    apple_orange_count = int(fruit_string)
    
    # Calculate the number of mango fruits
    mango_count = total_fruits - apple_orange_count

    return mango_count

# Test case
# Let's say the string "8" represents 8 apples and oranges combined, and the total number of fruits is 12
# Therefore, there should be 4 mangoes in the basket
fruit_string = "8"
total_fruits = 12
print(method(fruit_string, total_fruits))  # Output should be 4