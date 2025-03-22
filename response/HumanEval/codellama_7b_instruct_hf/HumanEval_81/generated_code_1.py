import numpy as np

def method():
    """
    This function takes a list of GPAs as input and outputs a list of letter grades using the following table:
    GPA | Letter grade
    4.0 A+ > 3.7 A > 3.3 A- > 3.0 B+ > 2.7 B > 2.3 B- > 2.0 C+ > 1.7 C > 1.3 C- > 1.0 D+ > 0.7 D > 0.0 D- 0.0 E
    """

    # Get the list of GPAs from the user
    # gpas = np.array(input("Enter the list of GPAs separated by spaces: ").split(), dtype=float)

    # 修改
    gpas = np.array([4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0])

    # Create a list of letter grades
    letter_grades = []
    for gpa in gpas:
        if gpa >= 4.0:
            letter_grades.append("A+")
        elif gpa >= 3.7:
            letter_grades.append("A")
        elif gpa >= 3.3:
            letter_grades.append("A-")
        elif gpa >= 3.0:
            letter_grades.append("B+")
        elif gpa >= 2.7:
            letter_grades.append("B")
        elif gpa >= 2.3:
            letter_grades.append("B-")
        elif gpa >= 2.0:
            letter_grades.append("C+")
        elif gpa >= 1.7:
            letter_grades.append("C")
        elif gpa >= 1.3:
            letter_grades.append("C-")
        elif gpa >= 1.0:
            letter_grades.append("D+")
        elif gpa >= 0.7:
            letter_grades.append("D")
        else:
            letter_grades.append("E")

    # Return the list of letter grades
    return letter_grades

# Test case
gpas = np.array([4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0])
letter_grades = method()
print(letter_grades)