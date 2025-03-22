def method(base, height):
    # Calculate the area of a triangle
    area = 0.5 * base * height
    return area

# Test case
def test_method():
    base = 10
    height = 5
    expected_area = 25.0  # Since 0.5 * 10 * 5 = 25.0
    assert method(base, height) == expected_area
    print("Test case passed.")

# Run the test case
test_method()