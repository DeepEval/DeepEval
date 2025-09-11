import re

class PersonRequest:
    """
    This class validates input personal information data and sets invalid fields to None based to specific rules.
    """

    def __init__(self, name: str, sex: str, phoneNumber: str):
        """
        Initialize PersonRequest object with the provided information.
        :param name: str, the name of the person
        :param sex: str, the sex of the person
        :param phoneNumber: str, the phone number of the person
        """
        self.name = self._validate_name(name)
        self.sex = self._validate_sex(sex)
        self.phoneNumber = self._validate_phoneNumber(phoneNumber)


    def _validate_name(self, name: str) -> str:
        """
        Validate the name and return it. If name is empty or exceeds 33 characters in length, set to None.
        :param name: str, the name to validate
        :return: str, the validated name or None if invalid
        """
        if not name or len(name) > 33:
            return None
        return name


    def _validate_sex(self, sex: str) -> str:
        """
        Validate the sex and return it. If sex is not Man, Woman, or UGM, set to None.
        :param sex: str, the sex to validate
        :return: str, the validated sex or None if invalid
        """
        valid_sex = ["Man", "Woman", "UGM"]
        if sex in valid_sex:
            return sex
        return None


    def _validate_phoneNumber(self, phoneNumber: str) -> str:
        """
        Validate the phone number and return it. If phoneNumber is empty or not an 11 digit number, set to None.
        :param phoneNumber: str, the phone number to validate
        :return: str, the validated phone number or None if invalid
        """
        pattern = re.compile(r'^\d{11}$')
        if not phoneNumber or not pattern.match(phoneNumber):
            return None
        return phoneNumber


if __name__ == "__main__":
    instance = PersonRequest("John Doe", "Man", "12345678901")
    print(f"Name: {instance.name}, Sex: {instance.sex}, PhoneNumber: {instance.phoneNumber}")

    # Test case for _validate_name method
    instance = PersonRequest("", "Man", "12345678901")
    print(f"Name: {instance.name}, Sex: {instance.sex}, PhoneNumber: {instance.phoneNumber}")

    instance = PersonRequest("John Doe John Doe John Doe", "Man", "12345678901")
    print(f"Name: {instance.name}, Sex: {instance.sex}, PhoneNumber: {instance.phoneNumber}")

    # Test case for _validate_sex method
    instance = PersonRequest("John Doe", "Other", "12345678901")
    print(f"Name: {instance.name}, Sex: {instance.sex}, PhoneNumber: {instance.phoneNumber}")

    # Test case for _validate_phoneNumber method
    instance = PersonRequest("John Doe", "Man", "")
    print(f"Name: {instance.name}, Sex: {instance.sex}, PhoneNumber: {instance.phoneNumber}")

    instance = PersonRequest("John Doe", "Man", "1234567890")
    print(f"Name: {instance.name}, Sex: {instance.sex}, PhoneNumber: {instance.phoneNumber}")
    instance = PersonRequest("John Doe", "Man", "1234567890123")
    print(f"Name: {instance.name}, Sex: {instance.sex}, PhoneNumber: {instance.phoneNumber}")