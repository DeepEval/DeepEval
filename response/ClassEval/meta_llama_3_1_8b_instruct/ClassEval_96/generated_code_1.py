import math

class WeatherSystem:
    """
    This is a class representing a weather system that provides functionality to query weather information for a specific city and convert temperature units between Celsius and Fahrenheit.
    """

    def __init__(self, city) -> None:
        """
        Initialize the weather system with a city name.
        """
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}

    def query(self, weather_list, tmp_units='celsius'):
        """
        Query the weather system for the weather and temperature of the city, and convert the temperature units based on the input parameter.
        :param weather_list: a dictionary of weather information for different cities, dict.
        :param tmp_units: the temperature units to convert to, str.
        :return: the temperature and weather of the city, tuple.
        """
        if self.city in weather_list:
            self.weather = weather_list[self.city]['weather']
            self.temperature = weather_list[self.city]['temperature']
            if weather_list[self.city]['temperature units']!= tmp_units:
                if tmp_units == 'celsius':
                    self.temperature = (self.temperature - 32) * 5 / 9
                elif tmp_units == 'fahrenheit':
                    self.temperature = (self.temperature * 9 / 5) + 32
            return (self.temperature, self.weather)
        else:
            return None

    def set_city(self, city):
        """
        Set the city of the weather system.
        :param city: the city to set, str.
        :return: None
        """
        self.city = city

    def celsius_to_fahrenheit(self):
        """
        Convert the temperature from Celsius to Fahrenheit.
        :return: the temperature in Fahrenheit, float.
        """
        if self.temperature is not None:
            return (self.temperature * 9 / 5) + 32
        else:
            return None

    def fahrenheit_to_celsius(self):
        """
        Convert the temperature from Fahrenheit to Celsius.
        :return: the temperature in Celsius, float.
        """
        if self.temperature is not None:
            return (self.temperature - 32) * 5 / 9
        else:
            return None

if __name__ == "__main__":
    weatherSystem = WeatherSystem('New York')
    weather_list = {'New York': {'weather':'sunny', 'temperature': 27, 'temperature units': 'celsius'}, 
                    'Beijing': {'weather': 'cloudy', 'temperature': 23, 'temperature units': 'celsius'}}
    
    # Test case for query method
    print(weatherSystem.query(weather_list, 'celsius'))  # Output: (27.0,'sunny')
    print(weatherSystem.query(weather_list, 'fahrenheit'))  # Output: (80.6,'sunny')
    
    # Test case for set_city method
    weatherSystem.set_city('Beijing')
    print(weatherSystem.city)  # Output: Beijing
    
    # Test case for celsius_to_fahrenheit method
    weatherSystem.temperature = 27
    print(weatherSystem.celsius_to_fahrenheit())  # Output: 80.6
    
    # Test case for fahrenheit_to_celsius method
    weatherSystem.temperature = 80.6
    print(weatherSystem.fahrenheit_to_celsius())  # Output: 26.999999999999996