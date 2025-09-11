import math

class FitnessTracker:
    def __init__(self, height, weight, age, sex):
        self.height = height
        self.weight = weight
        self.age = age
        self.sex = sex
        self.BMI_std = [{"male": [20, 25]}, {"female": [19, 24]}]

    def get_BMI(self):
        BMI = self.weight / (self.height ** 2)
        return BMI

    def condition_judge(self):
        if self.sex == "male":
            if self.BMI >= self.BMI_std["male"][0] and self.BMI <= self.BMI_std["male"][1]:
                return 0
            elif self.BMI < self.BMI_std["male"][0]:
                return -1
            else:
                return 1
        elif self.sex == "female":
            if self.BMI >= self.BMI_std["female"][0] and self.BMI <= self.BMI_std["female"][1]:
                return 0
            elif self.BMI < self.BMI_std["female"][0]:
                return -1
            else:
                return 1

    def calculate_calorie_intake(self):
        BMR = self.get_BMR()
        if self.condition_judge() == -1:
            calorie_intake = BMR * 1.2
        elif self.condition_judge() == 1:
            calorie_intake = BMR * 1.6
        else:
            calorie_intake = BMR * 1.4
        return calorie_intake

    def get_BMR(self):
        if self.sex == "male":
            BMR = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        elif self.sex == "female":
            BMR = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        return BMR

if __name__ == "__main__":
    ft = FitnessTracker(1.8, 70, 20, "male")
    print(ft.get_BMI())
    print(ft.condition_judge())
    print(ft.calculate_calorie_intake())
    print(ft.get_BMR())