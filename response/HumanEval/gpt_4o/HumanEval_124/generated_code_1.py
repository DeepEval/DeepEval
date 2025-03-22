import re

def method(date_string):
    # Check if the date string is empty
    if not date_string:
        return False

    # Define a regex pattern to match the date format mm-dd-yyyy
    date_pattern = r'^(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])-(\d{4})$'
    match = re.match(date_pattern, date_string)

    if not match:
        return False

    # Extract month, day, and year from the regex groups
    month = int(match.group(1))
    day = int(match.group(2))
    year = int(match.group(3))

    # Check the number of days per month
    if month in {1, 3, 5, 7, 8, 10, 12}:
        return 1 <= day <= 31
    elif month in {4, 6, 9, 11}:
        return 1 <= day <= 30
    elif month == 2:
        # Check for leap year if day is 29
        if day == 29:
            # Leap year check: Year divisible by 4 and (not divisible by 100 or divisible by 400)
            return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        return 1 <= day <= 28
    else:
        return False

# Test case for validation
print(method("02-29-2024"))  # Expected output: True (2024 is a leap year)
print(method("02-29-2023"))  # Expected output: False (2023 is not a leap year)
print(method("04-31-2023"))  # Expected output: False (April has 30 days)
print(method("06-30-2023"))  # Expected output: True (June has 30 days)
print(method("13-01-2023"))  # Expected output: False (Invalid month)
print(method("11-00-2023"))  # Expected output: False (Invalid day)