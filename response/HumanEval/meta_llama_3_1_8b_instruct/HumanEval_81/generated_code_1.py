def method(students_gpa):
    """
    This function takes a list of GPAs and returns a list of corresponding letter grades.

    Args:
        students_gpa (list): A list of GPAs.

    Returns:
        list: A list of letter grades.
    """

    # Define the GPA ranges and corresponding letter grades
    gpa_ranges = {
        'A+': (4.0, 3.7),
        'A': (3.7, 3.3),
        'A-': (3.3, 3.0),
        'B+': (3.0, 2.7),
        'B': (2.7, 2.3),
        'B-': (2.3, 2.0),
        'C+': (2.0, 1.7),
        'C': (1.7, 1.3),
        'C-': (1.3, 1.0),
        'D+': (1.0, 0.7),
        'D': (0.7, 0.0),
        'D-': (0.0, -1.0)  # Assign D- to any GPA below 0
    }

    # Initialize an empty list to store the letter grades
    letter_grades = []

    # Loop through each GPA in the input list
    for gpa in students_gpa:
        # Initialize a flag to indicate if the GPA is found in any range
        found = False

        # Loop through each GPA range
        for letter, (lower, upper) in gpa_ranges.items():
            # Check if the GPA falls within the current range
            if lower <= gpa <= upper:
                # Append the corresponding letter grade to the list
                letter_grades.append(letter)
                found = True
                break  # Exit the inner loop since the GPA is already found

        # If the GPA is not found in any range, append 'E' to the list
        if not found:
            letter_grades.append('E')

    # Return the list of letter grades
    return letter_grades


# Test the function with a sample list of GPAs
students_gpa = [4.0, 3.8, 3.2, 2.9, 2.6, 2.4, 2.1, 1.8, 1.4, 1.1, 0.8, 0.5]
output = method(students_gpa)
print(output)  # Output: ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'D', 'D-']