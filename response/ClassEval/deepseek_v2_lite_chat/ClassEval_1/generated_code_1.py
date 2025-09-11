import math

class AreaCalculator:
    """
    This is a class for calculating the area of different shapes, including circle, sphere, cylinder, sector and annulus.
    """

    def __init__(self, radius):
        """
        Initialize the radius for shapes.
        :param radius: float
        """
        self.radius = radius

    def calculate_circle_area(self):
        """
        Calculate the area of a circle based on the radius.
        :return: Area of the circle, float
        """
        return math.pi * self.radius ** 2

    def calculate_sphere_area(self):
        """
        Calculate the surface area of a sphere based on the radius.
        :return: Surface area of the sphere, float
        """
        return 4 * math.pi * self.radius ** 2

    def calculate_cylinder_area(self, height):
        """
        Calculate the lateral surface area of a cylinder based on the radius and height.
        :param height: Height of the cylinder, float
        :return: Lateral surface area of the cylinder, float
        """
        return 2 * math.pi * self.radius * height

    def calculate_sector_area(self, angle):
        """
        Calculate the area of a sector based on the radius and angle.
        :param angle: Angle of the sector, in radians
        :return: Area of the sector, float
        """
        return 0.5 * self.radius ** 2 * angle

    def calculate_annulus_area(self, inner_radius, outer_radius):
        """
        Calculate the area of an annulus (a ring shape) based on the inner and outer radii.
        :param inner_radius: Inner radius of the annulus, float
        :param outer_radius: Outer radius of the annulus, float
        :return: Area of the annulus, float
        """
        return math.pi * (outer_radius ** 2 - inner_radius ** 2)

# Test cases to validate the methods
if __name__ == "__main__":
    # Create an instance of AreaCalculator
    area_calculator = AreaCalculator(2)
    
    # Test cases
    print("Circle area:", area_calculator.calculate_circle_area())
    print("Sphere surface area:", area_calculator.calculate_sphere_area())
    print("Cylinder lateral area:", area_calculator.calculate_cylinder_area(3))
    print("Sector area:", area_calculator.calculate_sector_area(math.pi))
    print("Annulus area:", area_calculator.calculate_annulus_area(2, 3))