import math

planets = {
    "Mercury": 0.39,
    "Venus": 0.72,
    "Earth": 1,
    "Mars": 1.52,
    "Jupiter": 5.20,
    "Saturn": 9.54,
    "Uranus": 19.19,
    "Neptune": 30.07,
}

def get_planets_between(planet1, planet2):
    """
    Returns a tuple containing all planets whose orbits are located between the orbit of planet1 and the orbit of planet2, sorted by the proximity to the sun. The function should return an empty tuple if planet1 or planet2 are not correct planet names.
    """

    if planet1 not in planets or planet2 not in planets:
        return ()

    index1 = list(planets.keys()).index(planet1)
    index2 = list(planets.keys()).index(planet2)

    if index1 > index2:
        planet1, planet2 = planet2, planet1
        index1, index2 = index2, index1

    planets_between = list(planets.keys())[index1 + 1 : index2]
    planets_between.sort(key=planets.get)

    return tuple(planets_between)

# Test case
planets_between = get_planets_between("Mercury", "Neptune")
print(planets_between)  # Output: ('Venus', 'Earth', 'Mars')