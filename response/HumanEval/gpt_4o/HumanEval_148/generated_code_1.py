def method(planet1, planet2):
    # List of planets ordered by proximity to the Sun
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    # Check if both planet1 and planet2 are valid planet names
    if planet1 not in planets or planet2 not in planets:
        return tuple()
    
    # Find the indices of the two planets
    index1 = planets.index(planet1)
    index2 = planets.index(planet2)
    
    # Ensure index1 is less than index2 for easier slicing
    if index1 > index2:
        index1, index2 = index2, index1
    
    # Return the planets between the two given planets
    output = tuple(planets[index1 + 1:index2])
    
    return output

# Test case for validation
test_output = method('Earth', 'Saturn')
print(test_output)  # Expected output: ('Mars', 'Jupiter')