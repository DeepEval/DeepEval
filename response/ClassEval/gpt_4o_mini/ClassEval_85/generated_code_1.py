import time

class Thermostat:
    """
    The class manages temperature control, including setting and retrieving the target temperature,
    adjusting the mode, and simulating temperature operation.
    """

    def __init__(self, current_temperature, target_temperature, mode):
        """
        Initialize instances of the Thermostat class, including the current temperature, target temperature, and operating mode.
        :param current_temperature: float
        :param target_temperature: float
        :param mode: str, the work mode
        """
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
        self.mode = mode

    def get_target_temperature(self):
        """
        Get the target temperature of an instance of the Thermostat class.
        :return: float
        """
        return self.target_temperature

    def set_target_temperature(self, temperature):
        """
        Set the target temperature.
        :param temperature: float, the target temperature
        """
        self.target_temperature = temperature

    def get_mode(self):
        """
        Get the current work mode.
        :return: str, working mode. only ['heat', 'cool']
        """
        return self.mode

    def set_mode(self, mode):
        """
        Set the current work mode.
        :param mode: str, working mode. only ['heat', 'cool']
        """
        if mode in ['heat', 'cool']:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'heat' or 'cool'.")

    def auto_set_mode(self):
        """
        Automatically set the operating mode by comparing with the current temperature and target temperature.
        If the current temperature is lower than the target temperature, the operating mode is set to 'heat',
        otherwise it is set to 'cool'.
        """
        if self.current_temperature < self.target_temperature:
            self.mode = 'heat'
        else:
            self.mode = 'cool'

    def auto_check_conflict(self):
        """
        Check if there is a conflict between the operating mode and the relationship between the current temperature and the target temperature.
        If there is a conflict, the operating mode will be adjusted automatically.
        :return: bool, True if mode isn't conflict with the relationship between the current temperature and the target temperature, or False otherwise.
        """
        if (self.current_temperature < self.target_temperature and self.mode != 'heat') or \
           (self.current_temperature >= self.target_temperature and self.mode != 'cool'):
            self.auto_set_mode()
            return False
        return True

    def simulate_operation(self):
        """
        Simulate the operation of Thermostat. It will automatically start the auto_set_mode method to set the operating mode,
        and then automatically adjust the current temperature according to the operating mode until the target temperature is reached.
        :return: float, the time it took to complete the simulation.
        """
        start_time = time.time()
        self.auto_set_mode()

        while self.current_temperature != self.target_temperature:
            if self.mode == 'heat':
                self.current_temperature += 1  # Increase temperature
            else:
                self.current_temperature -= 1  # Decrease temperature
            time.sleep(1)  # Simulate the time taken to adjust the temperature

        end_time = time.time()
        return end_time - start_time

# Test cases
if __name__ == "__main__":
    thermostat = Thermostat(20.4, 37.5, 'cool')

    # Test get_target_temperature
    print(thermostat.get_target_temperature())  # Output: 37.5

    # Test set_target_temperature
    thermostat.set_target_temperature(37.6)
    print(thermostat.target_temperature)  # Output: 37.6

    # Test get_mode
    print(thermostat.get_mode())  # Output: 'cool'

    # Test set_mode
    thermostat.set_mode('heat')
    print(thermostat.mode)  # Output: 'heat'

    # Test auto_set_mode
    thermostat.auto_set_mode()
    print(thermostat.mode)  # Output: 'heat'

    # Test auto_check_conflict
    print(thermostat.auto_check_conflict())  # Output: False
    print(thermostat.mode)  # Output: 'heat' (no conflict, should remain 'heat')

    # Test simulate_operation
    # time_taken = thermostat.simulate_operation()
    # print(f"Time taken to reach target temperature: {time_taken:.2f} seconds")
    # print(thermostat.current_temperature)  # Output: 37.5