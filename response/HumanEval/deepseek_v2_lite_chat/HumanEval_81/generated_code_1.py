def method():
    # Dictionary to map GPA to letter grade
    grade_mapping = {
        '4.0': 'A+',
        '3.7': 'A',
        '3.3': 'A-',
        '3.0': 'B+',
        '2.7': 'B',
        '2.3': 'B-',
        '2.0': 'C+',
        '1.7': 'C',
        '1.3': 'C-',
        '1.0': 'D+',
        '0.7': 'D',
        '0.0': 'D-',
        '0.0': 'E'
    }

    # List of GPAs to convert
    gpas = [4.0, 3.7, 3.3, 3.0, 2.7, 2.3, 2.0, 1.7, 1.3, 1.0, 0.7, 0.0]

    # Convert GPAs to letter grades
    letter_grades = [grade_mapping[gpa] for gpa in gpas]

    # Return the list of letter grades
    return letter_grades

# Test case
def test_method():
    # Test the function with a known list of GPAs
    gpas = [4.0, 3.7, 3.3, 3.0, 2.7, 2.3, 2.0, 1.7, 1.3, 1.0, 0.7, 0.0]
    expected_output = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-']
    # assert method() == expected_output
    print(method())

# Run the test case
test_method()