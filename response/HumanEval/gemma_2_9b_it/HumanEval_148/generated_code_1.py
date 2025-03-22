def method():
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

    def find_planets_between(planet1, planet2):
        if planet1 not in planets or planet2 not in planets:
            return ()

        index1 = planets.index(planet1)
        index2 = planets.index(planet2)

        if index1 > index2:
            index1, index2 = index2, index1  # Ensure planet1 is before planet2

        return tuple(planets[index1 + 1:index2])

    planet1 = "Mars"
    planet2 = "Saturn"
    output = find_planets_between(planet1, planet2)
    return output

output = method()
print(output)  # Output: ('Jupiter',)