from datetime import datetime

def validate_date(date_string):
    """
    Validate a given date string.

    Args:
    date_string (str): The date string to be validated in the format mm-dd-yyyy.

    Returns:
    bool: True if the date is valid, False otherwise.
    """
    # Check if the date string is not empty
    if not date_string:
        return False

    # Try to parse the date string
    try:
        # Parse the date string into a datetime object
        date = datetime.strptime(date_string, '%m-%d-%Y')
        
        # Check if the month is valid (1-12)
        if date.month < 1 or date.month > 12:
            return False
        
        # Check if the day is valid for the given month
        if date.month in [1, 3, 5, 7, 8, 10, 12]:
            # Months with 31 days
            if date.day < 1 or date.day > 31:
                return False
        elif date.month in [4, 6, 9, 11]:
            # Months with 30 days
            if date.day < 1 or date.day > 30:
                return False
        else:
            # February
            if date.day < 1 or date.day > 29:
                return False
        
        # If all checks pass, the date is valid
        return True
    
    # If parsing fails, the date string is invalid
    except ValueError:
        return False


# Test case
date_string = '07-25-2022'
print(validate_date(date_string))  # Expected output: True

date_string = '02-29-2020'
print(validate_date(date_string))  # Expected output: True

date_string = '02-30-2020'
print(validate_date(date_string))  # Expected output: False

date_string = '01-01-2020'
print(validate_date(date_string))  # Expected output: True

date_string = ''
print(validate_date(date_string))  # Expected output: False