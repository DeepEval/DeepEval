class FitnessTracker:
    """
    This is a class as a fitness tracker that implements to calculate BMI (Body Mass Index) 
    and calorie intake based on the user's height, weight, age, and sex.
    """

    def __init__(self, height, weight, age, sex) -> None:
        """
        Initialize the class with height, weight, age, and sex, and calculate the BMI standard 
        based on sex, where male is 20-25 and female is 19-24.
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
        return self.weight / (self.height ** 2)

    def condition_judge(self):
        """
        Judge the condition of the user based on the BMI standard.
        :return: 1 if the user is too fat, -1 if the user is too thin, 0 if the user is normal, int.
        >>> fitnessTracker = FitnessTracker(1.8, 70, 20, "male")
        >>> fitnessTracker.condition_judge()
        0
        """
        bmi = self.get_BMI()
        if self.sex not in self.BMI_std:
            raise ValueError("Invalid sex provided.")
        
        bmi_range = self.BMI_std[self.sex]
        if bmi < bmi_range[0]:
            return -1  # Too thin
        elif bmi > bmi_range[1]:
            return 1   # Too fat
        else:
            return 0   # Normal

    def calculate_calorie_intake(self):
        """
        Calculate the calorie intake based on the user's condition and BMR (Basal Metabolic Rate),
        BMR is calculated based on the user's height, weight, age, and sex,
        male: 10 * self.weight + 6.25 * self.height - 5 * self.age + 5,
        female: 10 * self.weight + 6.25 * self.height - 5 * self.age - 161.
        The calorie intake is calculated based on the BMR and the user's condition:
        if the user is too fat, the calorie intake is BMR * 1.2,
        if the user is too thin, the calorie intake is BMR * 1.6,
        if the user is normal, the calorie intake is BMR * 1.4.
        :return: calorie intake, float.
        >>> fitnessTracker = FitnessTracker(1.8, 70, 20, "male")
        >>> fitnessTracker.calculate_calorie_intake()
        1686.0
        """
        if self.sex == "male":
            bmr = 10 * self.weight + 6.25 * self.height * 100 - 5 * self.age + 5  # height in cm
        elif self.sex == "female":
            bmr = 10 * self.weight + 6.25 * self.height * 100 - 5 * self.age - 161  # height in cm
        else:
            raise ValueError("Invalid sex provided.")

        condition = self.condition_judge()
        if condition == 1:  # Too fat
            calorie_intake = bmr * 1.2
        elif condition == -1:  # Too thin
            calorie_intake = bmr * 1.6
        else:  # Normal
            calorie_intake = bmr * 1.4
        
        return calorie_intake

# Test cases to validate the code executability
if __name__ == "__main__":
    # Test for get_BMI
    fitnessTracker = FitnessTracker(1.8, 70, 20, "male")
    bmi = fitnessTracker.get_BMI()
    print(f"BMI: {bmi}")  # Expected output: 21.604938271604937

    # Test for condition_judge
    condition = fitnessTracker.condition_judge()
    print(f"Condition: {condition}")  # Expected output: 0

    # Test for calculate_calorie_intake
    calorie_intake = fitnessTracker.calculate_calorie_intake()
    print(f"Calorie Intake: {calorie_intake}")  # Expected output: 1686.0