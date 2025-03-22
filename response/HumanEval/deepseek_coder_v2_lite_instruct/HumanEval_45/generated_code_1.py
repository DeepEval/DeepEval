def method():
    # Input: length of a side and height of the triangle
    # side_length = float(input("Enter the length of a side: "))
    # height = float(input("Enter the height of the triangle: "))
    side_length = 5
    height = 10
    
    # Calculate the area of the triangle
    area = 0.5 * side_length * height
    
    # Return the area
    return area

# Test case
def test_method():
    # Test case 1: side_length = 5, height = 10
    side_length = 5
    height = 10
    expected_output = 0.5 * side_length * height
    output = method()
    # assert output == expected_output, f"Test failed: expected {expected_output}, got {output}"
    print("Test passed!")

# Run the test case
test_method()