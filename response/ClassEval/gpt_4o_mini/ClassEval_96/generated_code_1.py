class WeatherSystem:
    """
    This is a class representing a weather system that provides functionality to query weather information 
    for a specific city and convert temperature units between Celsius and Fahrenheit.
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
        Query the weather system for the weather and temperature of the city,
        and convert the temperature units based on the input parameter.
        :param weather_list: a dictionary of weather information for different cities, dict.
        :param tmp_units: the temperature units to convert to, str.
        :return: the temperature and weather of the city, tuple.
        """
        if self.city in weather_list:
            weather_info = weather_list[self.city]
            self.weather = weather_info['weather']
            self.temperature = weather_info['temperature']
            
            if tmp_units == 'fahrenheit':
                self.temperature = self.celsius_to_fahrenheit()
                
            return self.temperature, self.weather
        else:
            raise ValueError(f"Weather information for {self.city} is not available.")

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


# Test cases to validate the functionality of the WeatherSystem class
if __name__ == "__main__":
    # Test case for WeatherSystem initialization
    weatherSystem = WeatherSystem('New York')
    print(weatherSystem.city)  # Output: New York

    # Test case for query method
    weather_list = {
        'New York': {'weather': 'sunny', 'temperature': 27, 'temperature units': 'celsius'},
        'Beijing': {'weather': 'cloudy', 'temperature': 23, 'temperature units': 'celsius'}
    }
    output = weatherSystem.query(weather_list)
    print(output)  # Output: (27, 'sunny')

    # Test case for set_city method
    weatherSystem.set_city('Beijing')
    print(weatherSystem.city)  # Output: Beijing

    # Test case for celsius_to_fahrenheit method
    weatherSystem.temperature = 27
    fahrenheit_output = weatherSystem.celsius_to_fahrenheit()
    print(fahrenheit_output)  # Output: 80.6

    # Test case for fahrenheit_to_celsius method
    weatherSystem.temperature = 80.6
    celsius_output = weatherSystem.fahrenheit_to_celsius()
    print(celsius_output)  # Output: 27.0