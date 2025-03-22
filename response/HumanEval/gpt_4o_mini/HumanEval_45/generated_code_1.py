def method(base, height):
    # Calculate the area of the triangle
    area = 0.5 * base * height
    return area

# Test case
if __name__ == "__main__":
    base = 5  # length of the side
    height = 10  # height from the opposite vertex
    output = method(base, height)
    print(f"The area of the triangle with base {base} and height {height} is: {output}")