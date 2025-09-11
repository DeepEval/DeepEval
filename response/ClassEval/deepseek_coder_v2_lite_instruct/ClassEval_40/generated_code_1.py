class FitnessTracker:
    """
    This is a class as fitness tracker that implements to calculate BMI (Body Mass Index) and calorie intake based on the user's height, weight, age, and sex.
    """

    def __init__(self, height, weight, age, sex) -> None:
        """
        Initialize the class with height, weight, age, and sex, and calculate the BMI standard based on sex, and male is 20-25, female is 19-24.
        """
        self.height = height
        self.weight = weight
        self.age = age
        self.sex = sex
        self.BMI_std = [
            {"male": [20, 25]},
            {"female": [19, 24]}
        ]

    def get_BMI(self):
        """
        Calculate the BMI based on the height and weight.
        :return: BMI, which is the weight divide by the square of height, float.
        """
        bmi = self.weight / (self.height ** 2)
        return bmi

    def condition_judge(self):
        """
        Judge the condition of the user based on the BMI standard.
        :return: 1 if the user is too fat, -1 if the user is too thin, 0 if the user is normal, int.
        """
        bmi = self.get_BMI()
        if self.sex == "male":
            std = self.BMI_std[0]["male"]
        else:
            std = self.BMI_std[1]["female"]
        
        if bmi < std[0]:
            return -1
        elif bmi > std[1]:
            return 1
        else:
            return 0

    def calculate_calorie_intake(self):
        """
        Calculate the calorie intake based on the user's condition and BMR (Basal Metabolic Rate),
        BMR is calculated based on the user's height, weight, age, and sex,
        male is 10 * self.weight + 6.25 * self.height - 5 * self.age + 5,
        female is 10 * self.weight + 6.25 * self.height - 5 * self.age - 161,
        and the calorie intake is calculated based on the BMR and the user's condition,
        if the user is too fat, the calorie intake is BMR * 1.2,
        if the user is too thin, the calorie intake is BMR * 1.6,
        if the user is normal, the calorie intake is BMR * 1.4.
        :return: calorie intake, float.
        """
        bmr = None
        if self.sex == "male":
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        
        condition = self.condition_judge()
        if condition == 1:
            return bmr * 1.2
        elif condition == -1:
            return bmr * 1.6
        else:
            return bmr * 1.4

if __name__ == "__main__":
    # Test cases
    fitnessTracker = FitnessTracker(1.8, 70, 20, "male")
    print("BMI:", fitnessTracker.get_BMI())
    print("Condition:", fitnessTracker.condition_judge())
    print("Calorie Intake:", fitnessTracker.calculate_calorie_intake())