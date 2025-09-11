import math

class AreaCalculator:
    """
    This is a class for calculating the area of different shapes, including circle, sphere, cylinder, sector, and annulus.
    """

    def __init__(self, radius):
        """
        Initialize the radius for shapes.
        :param radius: float
        """
        self.radius = radius

    def calculate_circle_area(self):
        """
        Calculate the area of a circle based on self.radius.
        :return: area of circle, float
        """
        return math.pi * (self.radius ** 2)

    def calculate_sphere_area(self):
        """
        Calculate the surface area of a sphere based on self.radius.
        :return: surface area of sphere, float
        """
        return 4 * math.pi * (self.radius ** 2)

    def calculate_cylinder_area(self, height):
        """
        Calculate the surface area of a cylinder based on self.radius and height.
        :param height: height of cylinder, float
        :return: surface area of cylinder, float
        """
        return 2 * math.pi * self.radius * (self.radius + height)

    def calculate_sector_area(self, angle):
        """
        Calculate the area of a sector based on self.radius and angle.
        :param angle: angle of sector, float
        :return: area of sector, float
        """
        return 0.5 * (self.radius ** 2) * angle

    def calculate_annulus_area(self, inner_radius, outer_radius):
        """
        Calculate the area of an annulus based on inner_radius and outer_radius.
        :param inner_radius: inner radius of annulus, float
        :param outer_radius: outer radius of annulus, float
        :return: area of annulus, float
        """
        return math.pi * ((outer_radius ** 2) - (inner_radius ** 2))

if __name__ == "__main__":
    # Test case for circle area
    areaCalculator = AreaCalculator(2)
    output = areaCalculator.calculate_circle_area()
    print(f"Circle Area: {output}")  # Expected: 12.566370614359172

    # Test case for sphere area
    output = areaCalculator.calculate_sphere_area()
    print(f"Sphere Area: {output}")  # Expected: 50.26548245743669

    # Test case for cylinder area
    output = areaCalculator.calculate_cylinder_area(3)
    print(f"Cylinder Area: {output}")  # Expected: 62.83185307179586

    # Test case for sector area
    output = areaCalculator.calculate_sector_area(math.pi)
    print(f"Sector Area: {output}")  # Expected: 6.283185307179586

    # Test case for annulus area
    output = areaCalculator.calculate_annulus_area(2, 3)
    print(f"Annulus Area: {output}")  # Expected: 15.707963267948966