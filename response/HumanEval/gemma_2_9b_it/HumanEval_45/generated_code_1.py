def method():
    """
    Calculates the area of a triangle given the length of its side and height.

    Returns:
        float: The area of the triangle.
    """
    
    # side = float(input("Enter the length of a side of the triangle: "))
    # height = float(input("Enter the height of the triangle: "))

    # 修改为固定值
    side = 10
    height = 5

    area = 0.5 * side * height 

    return area

# Test case
if __name__ == "__main__":
    area = method()
    print(f"The area of the triangle is: {area}")