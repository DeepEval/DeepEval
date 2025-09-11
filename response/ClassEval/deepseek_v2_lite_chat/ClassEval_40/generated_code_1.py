import math

class FitnessTracker:
    """
    This is a class as fitness tracker that implements to calculate BMI (Body Mass Index) and calorie intake based on the user's height, weight, age, and sex.
    """

    def __init__(self, height, weight, age, sex) -> None:
        """
        Initialize the class with height, weight, age, and sex, and calculate the BMI standard based on sex, and male is 20-25, female is 19-24.
        """
        self.height = float(height)
        self.weight = float(weight)
        self.age = int(age)
        self.sex = sex
        self.BMI_std = {
            "male": [20, 25],
            "female": [19, 24]
        }

    def get_BMI(self):
        """
        Calculate the BMI based on the height and weight.
        :return: BMI, which is the weight divided by the square of height, float.
        """
        return self.weight / (self.height ** 2)

    def condition_judge(self):
        """
        Judge the condition of the user based on the BMI standard.
        :return: 1 if the user is too fat, -1 if the user is too thin, 0 if the user is normal, int.
        """
        bmi = self.get_BMI()
        for category, range in self.BMI_std.items():
            if bmi >= range[0] and bmi <= range[1]:
                return 1 if sex == "male" else -1 if sex == "female" else 0
        return 0

    def calculate_calorie_intake(self):
        """
        Calculate the calorie intake based on the user's condition and BMR (Basal Metabolic Rate).
        BMR is calculated based on the user's height, weight, age, and sex.
        Male BMR = 10 * weight + 6.25 * height - 5 * age + 50
        Female BMR = 10 * weight + 6.25 * height - 5 * age - 161
        Calorie intake is calculated based on the BMR and the user's condition.
        If the user is too fat, the calorie intake is BMR * 1.2.
        If the user is too thin, the calorie intake is BMR * 1.6.
        If the user is normal, the calorie intake is BMR * 1.4.
        :return: calorie intake, float.
        """
        bmr = 0
        if self.sex == "male":
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age + 50
        else:  # self.sex == "female"
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161

        if self.condition_judge() == 1:
            return bmr * 1.2
        elif self.condition_judge() == -1:
            return bmr * 1.6
        else:
            return bmr * 1.4


# Test cases
if __name__ == "__main__":
    fitness_tracker = FitnessTracker(1.8, 70, 20, "male")
    bmi = fitness_tracker.get_BMI()
    print("BMI:", bmi)
    condition = fitness_tracker.condition_judge()
    print("Condition:", condition)
    calorie_intake = fitness_tracker.calculate_calorie_intake()
    print("Calorie Intake:", calorie_intake)