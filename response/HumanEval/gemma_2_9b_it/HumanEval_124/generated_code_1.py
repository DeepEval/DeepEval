def method(date_string):
    """
    Validates a date string and returns True if valid, False otherwise.

    Args:
        date_string: The date string in the format "mm-dd-yyyy".

    Returns:
        True if the date is valid, False otherwise.
    """

    if not date_string:
        return False  

    try:
        month, day, year = map(int, date_string.split('-'))
    except ValueError:
        return False  

    if not 1 <= month <= 12:
        return False

    if month in [4, 6, 9, 11]:
        if not 1 <= day <= 30:
            return False
    elif month == 2:
        if not (1 <= day <= 29):  
            return False 
    else:
        if not 1 <= day <= 31:
            return False 

    return True


# Test Case
date1 = "02-29-2020"
date2 = "13-05-2023" 
date3 = "05-31-2023"
date4 = ""

print(f"Date '{date1}' is valid: {method(date1)}")
print(f"Date '{date2}' is valid: {method(date2)}")
print(f"Date '{date3}' is valid: {method(date3)}")
print(f"Date '{date4}' is valid: {method(date4)}")