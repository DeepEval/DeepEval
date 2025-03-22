def method():
    # Define the GPA to letter grade mapping
    grade_mapping = {
        4.0: 'A+',
        3.7: 'A',
        3.3: 'A-',
        3.0: 'B+',
        2.7: 'B',
        2.3: 'B-',
        2.0: 'C+',
        1.7: 'C',
        1.3: 'C-',
        1.0: 'D+',
        0.7: 'D',
        0.0: 'E'
    }

    # Example list of GPAs
    gpas = [4.0, 3.8, 2.5, 0.5, 3.5]

    # Convert GPAs to letter grades
    letter_grades = [grade_mapping.get(gpa, 'Invalid GPA') for gpa in gpas]

    # Return the list of letter grades
    return letter_grades

# Test the function
output = method()
print(output)  # Expected output: ['A+', 'A', 'C', 'E', 'B+']