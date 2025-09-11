import time

class Thermostat:
    def __init__(self, current_temperature, target_temperature, mode):
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
        self.mode = mode

    def get_target_temperature(self):
        return self.target_temperature

    def set_target_temperature(self, temperature):
        self.target_temperature = temperature

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = mode

    def auto_set_mode(self):
        if self.current_temperature < self.target_temperature:
            self.mode = 'heat'
        else:
            self.mode = 'cool'

    def auto_check_conflict(self):
        return self.mode in ['heat', 'cool']

    def simulate_operation(self):
        start_time = time.time()
        while True:
            self.auto_set_mode()
            self.current_temperature = self.current_temperature + 0.5 if self.mode == 'heat' else self.current_temperature - 0.5
            if self.current_temperature == self.target_temperature:
                end_time = time.time()
                return end_time - start_time

# Test cases
# if __name__ == "__main__":
#     # Test get_target_temperature
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     print(thermostat.get_target_temperature())  # Expected output: 37.5

#     # Test set_target_temperature
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     thermostat.set_target_temperature(37.6)
#     print(thermostat.get_target_temperature())  # Expected output: 37.6

#     # Test get_mode
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     print(thermostat.get_mode())  # Expected output: 'cool'

#     # Test set_mode
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     thermostat.set_mode('heat')
#     print(thermostat.get_mode())  # Expected output: 'heat'

#     # Test auto_set_mode
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     thermostat.auto_set_mode()
#     print(thermostat.get_mode())  # Expected output: 'heat'

#     # Test auto_check_conflict
#     thermostat = Thermostat(20.4, 37.5, 'heat')
#     print(thermostat.auto_check_conflict())  # Expected output: False

#     # Test simulate_operation
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     start_time = thermostat.simulate_operation()
#     end_time = thermostat.simulate_operation()
#     print(f"First simulation time: {start_time:.2f} seconds")
#     print(f"Second simulation time: {end_time:.2f} seconds")
    # Expected output: First simulation time and second simulation time will be different, but they should be close.