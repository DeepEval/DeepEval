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
        if self.mode == 'heat' and self.current_temperature < self.target_temperature:
            return True
        elif self.mode == 'cool' and self.current_temperature > self.target_temperature:
            return True
        else:
            return False

    def simulate_operation(self):
        start_time = time.time()
        while self.current_temperature != self.target_temperature:
            self.auto_set_mode()
            if self.auto_check_conflict():
                break
        end_time = time.time()
        return int(end_time - start_time)

if __name__ == "__main__":
    thermostat = Thermostat(20.4, 37.5, 'cool')
    print(thermostat.simulate_operation())