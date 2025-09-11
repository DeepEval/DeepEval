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
        :return: float, the target temperature
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
        :return: str, working mode. only ['heat', 'cool']
        """
        return self.mode

    def set_mode(self, mode):
        """
        Set the current work mode
        :param mode: str, working mode. only ['heat', 'cool']
        """
        if mode in ['heat', 'cool']:
            self.mode = mode

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
        if self.mode == 'heat' and self.current_temperature > self.target_temperature:
            self.mode = 'cool'
            return False
        elif self.mode == 'cool' and self.current_temperature < self.target_temperature:
            self.mode = 'heat'
            return False
        return True

    def simulate_operation(self):
        """
        simulate the operation of Thermostat. It will automatically start the auto_set_mode method to set the operating mode,
        and then automatically adjust the current temperature according to the operating mode until the target temperature is reached.
        :return: int, the time it took to complete the simulation.
        """
        start_time = time.time()
        while abs(self.current_temperature - self.target_temperature) > 0.5:
            if self.mode == 'heat':
                self.current_temperature += 0.5
            else:
                self.current_temperature -= 0.5
        return int(time.time() - start_time)

# if __name__ == "__main__":
#     # Test cases
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     print(thermostat.get_target_temperature())  # Test get_target_temperature
#     thermostat.set_target_temperature(37.6)
#     print(thermostat.get_target_temperature())  # Test set_target_temperature
#     print(thermostat.get_mode())  # Test get_mode
#     thermostat.set_mode('heat')
#     print(thermostat.get_mode())  # Test set_mode
#     thermostat.auto_set_mode()
#     print(thermostat.get_mode())  # Test auto_set_mode
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     print(thermostat.auto_check_conflict())  # Test auto_check_conflict
#     thermostat = Thermostat(20.4, 37.5, 'heat')
#     print(thermostat.auto_check_conflict())  # Test auto_check_conflict
#     thermostat = Thermostat(20.4, 37.5, 'cool')
#     print(thermostat.simulate_operation())  # Test simulate_operation