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
        self.BMI_std = {
            "male": [20, 25],
            "female": [19, 24]
        }

    def get_BMI(self):
        """
        Calculate the BMI based on the height and weight.
        :return: BMI, which is the weight divided by the square of height, float.
        >>> fitnessTracker = FitnessTracker(1.8, 70, 20, "male")
        >>> fitnessTracker.get_BMI()
        21.604938271604937
        """
        bmi = self.weight / (self.height ** 2)
        return bmi

    def condition_judge(self):
        """
        Judge the condition of the user based on the BMI standard.
        :return: 1 if the user is too fat, -1 if the user is too thin, 0 if the user is normal, int.
        >>> fitnessTracker = FitnessTracker(1.8, 70, 20, "male")
        >>> fitnessTracker.condition_judge()
        -1
        """
        bmi = self.get_BMI()
        lower_bound, upper_bound = self.BMI_std[self.sex]
        if bmi > upper_bound:
            return 1
        elif bmi < lower_bound:
            return -1
        else:
            return 0

    def calculate_calorie_intake(self):
        """
        Calculate the calorie intake based on the user's condition and BMR (Basal Metabolic Rate).
        :return: calorie intake, float.
        >>> fitnessTracker = FitnessTracker(1.8, 70, 20, "male")
        >>> fitnessTracker.calculate_calorie_intake()
        986.0
        """
        if self.sex == "male":
            bmr = 10 * self.weight + 6.25 * self.height * 100 - 5 * self.age + 5
        else:
            bmr = 10 * self.weight + 6.25 * self.height * 100 - 5 * self.age - 161

        condition = self.condition_judge()
        if condition == 1:
            calorie_intake = bmr * 1.2
        elif condition == -1:
            calorie_intake = bmr * 1.6
        else:
            calorie_intake = bmr * 1.4

        return calorie_intake

if __name__ == "__main__":
    # Test cases
    fitnessTracker = FitnessTracker(1.8, 70, 20, "male")
    
    # Test get_BMI method
    bmi = fitnessTracker.get_BMI()
    print(f"BMI: {bmi}")  # Expected: 21.604938271604937
    
    # Test condition_judge method
    condition = fitnessTracker.condition_judge()
    print(f"Condition: {condition}")  # Expected: -1
    
    # Test calculate_calorie_intake method
    calorie_intake = fitnessTracker.calculate_calorie_intake()
    print(f"Calorie Intake: {calorie_intake}")  # Expected: 986.0