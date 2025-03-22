import math

def method():
    # Function to check if a triangle is right-angled
    def is_right_angled(a, b, c):
        # Check if the lengths can form a triangle
        if a + b <= c or b + c <= a or a + c <= b:
            return False
        # Check if the triangle is right-angled using Pythagorean theorem
        return math.isclose(a**2 + b**2, c**2)
    
    # Example triangle lengths
    side_a = 3
    side_b = 4
    side_c = 5
    
    # Check if the triangle is right-angled
    result = is_right_angled(side_a, side_b, side_c)
    
    # Return the result
    return result

# Example usage
if __name__ == "__main__":
    result = method()
    print(result)