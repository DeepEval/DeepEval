import re

class PersonRequest:
    def __init__(self, name: str, sex: str, phoneNumber: str):
        self.name = self._validate_name(name)
        self.sex = self._validate_sex(sex)
        self.phoneNumber = self._validate_phoneNumber(phoneNumber)

    def _validate_name(self, name: str) -> str:
        if not name or len(name) > 33:
            return None
        return name

    def _validate_sex(self, sex: str) -> str:
        if sex not in ['Man', 'Woman', 'UGM']:
            return None
        return sex

    def _validate_phoneNumber(self, phoneNumber: str) -> str:
        if not phoneNumber or not re.match(r'^\d{11}$', phoneNumber):
            return None
        return phoneNumber

if __name__ == "__main__":
    # Test cases
    instance = PersonRequest("John Doe", "Man", "1234567890")
    print(instance.name, instance.sex, instance.phoneNumber)

    instance = PersonRequest("John Doe", "NonExistent", "1234567890")
    print(instance.name, instance.sex, instance.phoneNumber)

    instance = PersonRequest("", "Man", "1234567890")
    print(instance.name, instance.sex, instance.phoneNumber)

    instance = PersonRequest("John Doe", "UGM", "1234567890")
    print(instance.name, instance.sex, instance.phoneNumber)

    instance = PersonRequest("John Doe", "Man", "123456789")
    print(instance.name, instance.sex, instance.phoneNumber)

    instance = PersonRequest("John Doe", "Man", "1234567890123")
    print(instance.name, instance.sex, instance.phoneNumber)

    instance = PersonRequest("John Doe", "Man", "12345678901234567890")
    print(instance.name, instance.sex, instance.phoneNumber)