def method(gpas):
    # Initialize a list to hold the letter grades
    letter_grades = []
    
    # Iterate through each GPA and determine the corresponding letter grade
    for gpa in gpas:
        if gpa == 4.0:
            letter_grades.append("A+")
        elif gpa > 3.7:
            letter_grades.append("A")
        elif gpa > 3.3:
            letter_grades.append("A-")
        elif gpa > 3.0:
            letter_grades.append("B+")
        elif gpa > 2.7:
            letter_grades.append("B")
        elif gpa > 2.3:
            letter_grades.append("B-")
        elif gpa > 2.0:
            letter_grades.append("C+")
        elif gpa > 1.7:
            letter_grades.append("C")
        elif gpa > 1.3:
            letter_grades.append("C-")
        elif gpa > 1.0:
            letter_grades.append("D+")
        elif gpa > 0.7:
            letter_grades.append("D")
        elif gpa > 0.0:
            letter_grades.append("D-")
        else:
            letter_grades.append("E")
    
    return letter_grades

# Test case for validation
if __name__ == "__main__":
    test_gpas = [4.0, 3.8, 3.5, 2.9, 2.5, 1.9, 0.5]
    output = method(test_gpas)
    print(output)  # Expected output: ['A+', 'A', 'A-', 'B+', 'B-', 'D+', 'E']