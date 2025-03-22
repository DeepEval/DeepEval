def method(gpas):
  """
  This function takes a list of GPAs and returns a list of corresponding letter grades.

  Args:
      gpas: A list of numerical GPAs.

  Returns:
      A list of letter grades corresponding to the input GPAs.
  """

  grades = []
  for gpa in gpas:
    if gpa >= 4.0:
      grades.append('A+')
    elif gpa > 3.7:
      grades.append('A')
    elif gpa > 3.3:
      grades.append('A-')
    elif gpa > 3.0:
      grades.append('B+')
    elif gpa > 2.7:
      grades.append('B')
    elif gpa > 2.3:
      grades.append('B-')
    elif gpa > 2.0:
      grades.append('C+')
    elif gpa > 1.7:
      grades.append('C')
    elif gpa > 1.3:
      grades.append('C-')
    elif gpa > 1.0:
      grades.append('D+')
    elif gpa > 0.7:
      grades.append('D')
    elif gpa >= 0.0:
      grades.append('D-')
    else:
      grades.append('E')
  return grades

# Test Case
gpas = [3.9, 3.2, 2.8, 1.5, 0.6]
letter_grades = method(gpas)
print(f"GPAs: {gpas}")
print(f"Letter Grades: {letter_grades}")