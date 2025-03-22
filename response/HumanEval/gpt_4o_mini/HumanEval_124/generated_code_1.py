import re

def is_valid_date(date_str):
    # Check if the date string is empty
    if not date_str:
        return False

    # Check if the date string matches the format mm-dd-yyyy
    pattern = r'^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-(\d{4})$'
    match = re.match(pattern, date_str)
    if not match:
        return False

    # Split the date string into components
    month, day, year = map(int, date_str.split('-'))

    # Validate the month
    if month < 1 or month > 12:
        return False

    # Validate the day based on the month
    if month in [1, 3, 5, 7, 8, 10, 12]:  # Months with 31 days
        if day < 1 or day > 31:
            return False
    elif month in [4, 6, 9, 11]:  # Months with 30 days
        if day < 1 or day > 30:
            return False
    elif month == 2:  # February
        if day < 1 or day > 29:
            return False

    return True

def method():
    # Example test case
    test_date = "02-29-2020"  # Leap year date
    output = is_valid_date(test_date)
    return output

# Running the method to test the implementation
print(method())  # Should print True for a valid leap year date