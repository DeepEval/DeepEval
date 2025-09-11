import re

class PersonRequest:
    def __init__(self, name, sex, phone_number):
        self.name = self._validate_name(name)
        self.sex = self._validate_sex(sex)
        self.phone_number = self._validate_phone_number(phone_number)

    def _validate_name(self, name):
        if not name or len(name) > 33:
            return None
        return name

    def _validate_sex(self, sex):
        if sex not in ("Man", "Woman", "UGM"):
            return None
        return sex

    def _validate_phone_number(self, phone_number):
        if not phone_number or not re.match(r"^\d{11}$", phone_number):
            return None
        return phone_number
    
if __name__ == "__main__":
    # 测试代码
    person = PersonRequest("Alice", "Woman", "12345678901")

