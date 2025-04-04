def method(gpas):
    def gpa_to_letter(gpa):
        if gpa >= 4.0:
            return "A+"
        elif gpa > 3.7:
            return "A"
        elif gpa > 3.3:
            return "A-"
        elif gpa > 3.0:
            return "B+"
        elif gpa > 2.7:
            return "B"
        elif gpa > 2.3:
            return "B-"
        elif gpa > 2.0:
            return "C+"
        elif gpa > 1.7:
            return "C"
        elif gpa > 1.3:
            return "C-"
        elif gpa > 1.0:
            return "D+"
        elif gpa > 0.7:
            return "D"
        elif gpa > 0.0:
            return "D-"
        else:
            return "E"
    
    output = [gpa_to_letter(gpa) for gpa in gpas]
    return output

# Test case
test_gpas = [4.0, 3.8, 3.5, 2.9, 2.4, 1.8, 0.5]
print(method(test_gpas))  # Expected output: ['A+', 'A', 'A-', 'B', 'B-', 'C', 'E']