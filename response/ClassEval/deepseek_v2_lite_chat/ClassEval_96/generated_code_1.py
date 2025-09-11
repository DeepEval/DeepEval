import requests
from datetime import datetime

class WeatherSystem:
    """
    This is a class representing a weather system that provides functionality to query weather information for a specific city and convert temperature units between Celsius and Fahrenheit.
    """

    def __init__(self, city, api_key):
        """
        Initialize the weather system with a city name and an API key.
        """
        self.city = city
        self.weather_list = {}
        self.api_key = api_key

    def query(self, weather_list, tmp_units='celsius'):
        """
        Query the weather system for the weather and temperature of the city, and convert the temperature units based on the input parameter.
        :param weather_list: a dictionary of weather information for different cities, dict.
        :param tmp_units: the temperature units to convert to, str.
        :return: the temperature and weather of the city, tuple.
        """
        # Query the weather API for the current weather
        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={self.city}&appid={self.api_key}&units={tmp_units}"
        )
        data = response.json()

        # Store the weather and temperature in the dictionary
        if self.weather_list.get(self.city):
            self.weather_list[self.city]['weather'] = data['weather'][0]['main']
            self.weather_list[self.city]['temperature'] = data['main'][tmp_units]
        else:
            self.weather_list[self.city] = {
                'weather': data['weather'][0]['main'],
                'temperature': data['main'][tmp_units]
            }

        return self.weather_list[self.city]['temperature'], self.weather_list[self.city]['weather']

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
            temp_fahrenheit = (self.temperature * 9/5) + 32
            return round(temp_fahrenheit, 2)
        else:
            return None

    def fahrenheit_to_celsius(self):
        """
        Convert the temperature from Fahrenheit to Celsius.
        :return: the temperature in Celsius, float.
        """
        if self.temperature is not None:
            temp_celsius = (self.temperature - 32) * 5/9
            return round(temp_celsius, 2)
        else:
            return None

# Example usage:
if __name__ == "__main__":
    weather_system = WeatherSystem('New York', 'your_api_key')
    weather_list = {'New York': {'weather': 'sunny', 'temperature': 27, 'temperature units': 'celsius'}}
    temperature, weather = weather_system.query(weather_list)
    print(f"Temperature: {temperature}°F, Weather: {weather}")
    print(f"Converted Temperature: {weather_system.celsius_to_fahrenheit()}°F, {weather_system.fahrenheit_to_celsius()}°C")