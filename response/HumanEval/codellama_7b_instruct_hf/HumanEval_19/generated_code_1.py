# Import necessary packages
import re

def method():
    # Get input string from user
    # input_str = input("Enter a string of numberals: ")

    # 修改：
    input_str = "two four one six three eight five nine zero"

    # Create a dictionary of numberals and their numerical values
    numberals = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}

    # Check if input string contains only valid numberals
    if not re.fullmatch(r"[zero|one|two|three|four|five|six|seven|eight|nine]+", input_str):
        raise ValueError("Input string contains invalid numberals.")

    # Convert numberals to numerical values
    numerical_values = [numberals[num] for num in input_str.split()]

    # Sort numerical values from smallest to largest
    numerical_values.sort()

    # Convert numerical values back to numberals
    sorted_numberals = [list(numberals.keys())[list(numberals.values()).index(num)] for num in numerical_values]

    # Join sorted numberals into a string
    output = " ".join(sorted_numberals)

    return output

# Test case
input_str = "two four one six three eight five nine zero"
output = method()
print(output)