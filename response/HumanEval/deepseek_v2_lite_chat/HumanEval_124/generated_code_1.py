import datetime

def method():
    def is_valid_date(year, month, day):
        # Check if day is valid for the given month and year
        if month in {1, 3, 5, 7, 8, 10, 12} and day < 1 or day > 31:
            return False
        elif month in {4, 6, 9, 11} and day < 1 or day > 30:
            return False
        elif month == 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) and day < 1 or day > 29:
            return False
        else:
            return True
    
    def validate_date_string(date_string):
        # Split the date string into year, month, day
        year, month, day = map(int, date_string.split('-'))
        # Create a datetime object
        date_object = datetime.date(year, month, day)
        # Check if the date is valid
        return is_valid_date(year, month, day)
    
    # Example date string to validate
    date_string = "02-29-2020"
    # Validate the date string
    output = validate_date_string(date_string)
    
    return output

# Test case
print(method())  # Expected output: True if date is valid, False otherwise