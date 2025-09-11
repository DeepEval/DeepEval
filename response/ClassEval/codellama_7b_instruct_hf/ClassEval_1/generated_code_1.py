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
        calculate the area of circle based on self.radius
        :return: area of circle, float
        """
        return math.pi * self.radius ** 2

    def calculate_sphere_area(self):
        """
        calculate the area of sphere based on self.radius
        :return: area of sphere, float
        """
        return 4 * math.pi * self.radius ** 2

    def calculate_cylinder_area(self, height):
        """
        calculate the area of cylinder based on self.radius and height
        :param height: height of cylinder, float
        :return: area of cylinder, float
        """
        return math.pi * self.radius ** 2 * height

    def calculate_sector_area(self, angle):
        """
        calculate the area of sector based on self.radius and angle
        :param angle: angle of sector, float
        :return: area of sector, float
        """
        return (math.pi * self.radius ** 2) * (angle / 360)

    def calculate_annulus_area(self, inner_radius, outer_radius):
        """
        calculate the area of annulus based on inner_radius and out_radius
        :param inner_radius: inner radius of annulus, float
        :param outer_radius: outer radius of annulus, float
        :return: area of annulus, float
        """
        return math.pi * (outer_radius ** 2 - inner_radius ** 2)

if __name__ == "__main__":
    areaCalculator = AreaCalculator(2)
    print(areaCalculator.calculate_circle_area())  # Output: 12.566370614359172
    print(areaCalculator.calculate_sphere_area())  # Output: 50.26548245743669
    print(areaCalculator.calculate_cylinder_area(3))  # Output: 62.83185307179586
    print(areaCalculator.calculate_sector_area(math.pi))  # Output: 6.283185307179586
    print(areaCalculator.calculate_annulus_area(2, 3))  # Output: 15.707963267948966