import requests

class WeatherSystem:
    def query(self, weather_list, tmp_units='celsius'):
        city = self.city
        if city not in weather_list:
            raise ValueError(f'Invalid city: {city}')

        weather_data = weather_list[city]
        temperature = weather_data['temperature']
        weather = weather_data['weather']

        if tmp_units == 'celsius':
            return temperature, weather
        elif tmp_units == 'fahrenheit':
            return (temperature * 9/5) + 32, weather
        else:
            raise ValueError(f'Invalid temperature unit: {tmp_units}')
        
if __name__ == '__main__':
    # Example usage
    weather_list = {
        'New York': {'temperature': 25, 'weather': 'Sunny'},
        'Los Angeles': {'temperature': 30, 'weather': 'Sunny'},
        'Chicago': {'temperature': 20, 'weather': 'Cloudy'}
    }
    
    ws = WeatherSystem()
    ws.city = 'New York'
    
    try:
        temperature, weather = ws.query(weather_list, tmp_units='celsius')
        print(f'Temperature: {temperature}°C, Weather: {weather}')
    except ValueError as e:
        print(e)