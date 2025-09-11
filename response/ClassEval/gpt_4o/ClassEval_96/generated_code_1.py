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

    def query(self, weather_list, tmp_units='celsius'):
        """
        Query the weather system for the weather and temperature of the city and convert the temperature units based on the input parameter.
        :param weather_list: a dictionary of weather information for different cities, dict.
        :param tmp_units: the temperature units to convert to, str.
        :return: the temperature and weather of the city, tuple.
        """
        if self.city in weather_list:
            self.weather = weather_list[self.city]['weather']
            self.temperature = weather_list[self.city]['temperature']

            # Convert temperature if needed
            if tmp_units.lower() == 'fahrenheit':
                self.temperature = self.celsius_to_fahrenheit()

            return self.temperature, self.weather
        else:
            return None, None

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
        return (self.temperature * 9/5) + 32

    def fahrenheit_to_celsius(self):
        """
        Convert the temperature from Fahrenheit to Celsius.
        :return: the temperature in Celsius, float.
        """
        return (self.temperature - 32) * 5/9


if __name__ == "__main__":
    # Test for query method
    weatherSystem = WeatherSystem('New York')
    weather_list = {
        'New York': {'weather': 'sunny', 'temperature': 27, 'temperature units': 'celsius'},
        'Beijing': {'weather': 'cloudy', 'temperature': 23, 'temperature units': 'celsius'}
    }
    output = weatherSystem.query(weather_list)
    print("Query Output:", output)  # Expected: (27, 'sunny')

    # Test for set_city method
    weatherSystem.set_city('Beijing')
    print("Set City Output:", weatherSystem.city)  # Expected: 'Beijing'

    # Test for celsius_to_fahrenheit method
    weatherSystem.temperature = 27
    fahrenheit = weatherSystem.celsius_to_fahrenheit()
    print("Celsius to Fahrenheit Output:", fahrenheit)  # Expected: 80.6

    # Test for fahrenheit_to_celsius method
    weatherSystem.temperature = 80.6
    celsius = weatherSystem.fahrenheit_to_celsius()
    print("Fahrenheit to Celsius Output:", celsius)  # Expected: 26.999999999999996