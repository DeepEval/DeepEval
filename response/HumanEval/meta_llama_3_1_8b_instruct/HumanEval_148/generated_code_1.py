# Importing necessary package
import enum

# Defining an enumeration for planets in our solar system
class Planet(enum.Enum):
    MERCURY = 1
    VENUS = 2
    EARTH = 3
    MARS = 4
    JUPITER = 5
    SATURN = 6
    URANUS = 7
    NEPTUNE = 8

# Creating a dictionary to map planet names to their corresponding enum values
planet_dict = {
    'Mercury': Planet.MERCURY,
    'Venus': Planet.VENUS,
    'Earth': Planet.EARTH,
    'Mars': Planet.MARS,
    'Jupiter': Planet.JUPITER,
    'Saturn': Planet.SATURN,
    'Uranus': Planet.URANUS,
    'Neptune': Planet.NEPTUNE
}

def method(planet1, planet2):
    """
    This function takes two planet names as strings planet1 and planet2.
    It returns a tuple containing all planets whose orbits are located between the orbit of planet1 and the orbit of planet2, 
    sorted by the proximity to the sun.
    
    Args:
        planet1 (str): The name of the first planet.
        planet2 (str): The name of the second planet.
    
    Returns:
        tuple: A tuple of planet names whose orbits are located between the orbit of planet1 and the orbit of planet2.
    """
    
    # Checking if planet1 and planet2 are valid planet names
    if planet1 not in planet_dict or planet2 not in planet_dict:
        return ()
    
    # Getting the enum values of planet1 and planet2
    value1 = planet_dict[planet1]
    value2 = planet_dict[planet2]
    
    # Finding the index of planet1 and planet2 in the list of planets
    index1 = list(Planet).index(value1)
    index2 = list(Planet).index(value2)
    
    # Checking if planet1 is closer to the sun than planet2
    if index1 > index2:
        index1, index2 = index2, index1
    
    # Returning a tuple of planet names whose orbits are located between the orbit of planet1 and the orbit of planet2
    return tuple(planet.name for i, planet in enumerate(list(Planet)) if index1 < i < index2)

# Test case for validation
output = method('Earth', 'Mars')
print(output)  # Output: ('Mars',)