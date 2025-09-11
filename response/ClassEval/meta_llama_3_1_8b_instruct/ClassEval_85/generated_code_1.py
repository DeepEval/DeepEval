import time

class Thermostat:
    """
    The class manages temperature control, including setting and retrieving the target temperature, adjusting the mode, and simulating temperature operation.
    """

    def __init__(self, current_temperature, target_temperature, mode):
        """
        initialize instances of the Thermostat class, including the current temperature, target temperature, and operating mode.
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
        :return self.current_temperature: int
        """
        return self.target_temperature

    def set_target_temperature(self, temperature):
        """
        Set the target temperature
        :param temperature: float, the target temperature
        """
        self.target_temperature = temperature

    def get_mode(self):
        """
        Get the current work mode
        :return mode: str, working mode. only ['heat', 'cool']
        """
        return self.mode

    def set_mode(self, mode):
        """
        Get the current work mode
        :param mode: str, working mode. only ['heat', 'cool']
        """
        if mode in ['heat', 'cool']:
            self.mode = mode
        else:
            raise ValueError("Invalid mode. Mode should be 'heat' or 'cool'.")

    def auto_set_mode(self):
        """
        Automatically set the operating mode by comparing with the current temperature and target temperature. If the current temperature is lower than the target temperature, the operating mode is set to 'heat', otherwise it is set to 'cool'.
        """
        if self.current_temperature < self.target_temperature:
            self.mode = 'heat'
        else:
            self.mode = 'cool'

    def auto_check_conflict(self):
        """
        Check if there is a conflict between the operating mode and the relationship between the current temperature and the target temperature.
        If there is a conflict, the operating mode will be adjusted automatically.
        :return: True if mode isn't conflict with the relationship between the current temperature and the target temperature, or False otherwise.
        """
        if (self.current_temperature < self.target_temperature and self.mode == 'cool') or \
           (self.current_temperature > self.target_temperature and self.mode == 'heat'):
            self.mode = 'heat' if self.current_temperature < self.target_temperature else 'cool'
            return False
        return True

    def simulate_operation(self):
        """
        simulate the operation of Thermostat. It will automatically start the auto_set_mode method to set the operating mode,
        and then automatically adjust the current temperature according to the operating mode until the target temperature is reached.
        :return time: int, the time it took to complete the simulation.
        """
        start_time = time.time()
        while abs(self.current_temperature - self.target_temperature) > 0.1:
            self.auto_set_mode()
            if self.mode == 'heat':
                self.current_temperature += 0.1
            else:
                self.current_temperature -= 0.1
            time.sleep(0.1)  # Simulate heating/cooling time
        end_time = time.time()
        return int(end_time - start_time)


if __name__ == "__main__":
    # Test case
    thermostat = Thermostat(20.4, 37.5, 'cool')
    print("Test get_target_temperature():")
    print(thermostat.get_target_temperature())  # Output: 37.5
    
    print("\nTest set_target_temperature():")
    thermostat.set_target_temperature(37.6)
    print(thermostat.target_temperature)  # Output: 37.6
    
    print("\nTest get_mode():")
    print(thermostat.get_mode())  # Output: cool
    
    print("\nTest set_mode():")
    thermostat.set_mode('cool')
    print(thermostat.mode)  # Output: cool
    
    print("\nTest auto_set_mode():")
    thermostat.auto_set_mode()
    print(thermostat.mode)  # Output: heat
    
    print("\nTest auto_check_conflict():")
    print(thermostat.auto_check_conflict())  # Output: False
    print(thermostat.mode)  # Output: heat
    
    print("\nTest simulate_operation():")
    print(thermostat.simulate_operation())  # Output: time it took to complete the simulation