import re
from datetime import datetime

def method():
    # Define the regular expression pattern for the date format
    date_pattern = r'^\d{2}-\d{2}-\d{4}$'
    
    # Get the date string from the user
    # date_string = input("Enter a date string in the format mm-dd-yyyy: ")
    date_string = "11-30-2024"
    
    # Check if the date string matches the pattern
    if not re.match(date_pattern, date_string):
        return False
    
    # Parse the date string
    try:
        date_object = datetime.strptime(date_string, "%m-%d-%Y")
    except ValueError:
        return False
    
    # Extract month, day, and year from the date object
    month = date_object.month
    day = date_object.day
    year = date_object.year
    
    # Check the number of days for each month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        if day < 1 or day > 31:
            return False
    elif month in [4, 6, 9, 11]:
        if day < 1 or day > 30:
            return False
    elif month == 2:
        if day < 1 or day > 29:
            return False
    else:
        return False
    
    # If all checks pass, return True
    return True

# Test case
print(method())  # Example test case