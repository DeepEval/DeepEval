class HRManagementSystem:
    def __init__(self):
        self.employees = {}

    def add_employee(self, employee_id, name, position, department, salary):
        if employee_id not in self.employees:
            self.employees[employee_id] = {'name': name, 'position': position, 'department': department, 'salary': salary}
            return True
        return False

    def remove_employee(self, employee_id):
        if employee_id in self.employees:
            del self.employees[employee_id]
            return True
        return False

    def update_employee(self, employee_id: int, employee_info: dict):
        if employee_id in self.employees:
            self.employees[employee_id] = {**self.employees[employee_id], **employee_info}
            return True
        return False

    def get_employee(self, employee_id):
        return self.employees.get(employee_id, False)

    def list_employees(self):
        return {employee_id: {k: v for k, v in employee.items()} for employee_id, employee in self.employees.items()}

if __name__ == "__main__":
    hr_system = HRManagementSystem()

    # Test case for add_employee
    print(hr_system.add_employee(1, 'John', 'Manager', 'Sales', 100000))  # True
    print(hr_system.add_employee(1, 'John', 'Manager', 'Sales', 100000))  # False

    # Test case for remove_employee
    hr_system.employees = {1: {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}}
    print(hr_system.remove_employee(1))  # True
    print(hr_system.remove_employee(2))  # False

    # Test case for update_employee
    hr_system.employees = {1: {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}}
    print(hr_system.update_employee(1, {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 20000}))  # True
    print(hr_system.update_employee(2, {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 20000}))  # False

    # Test case for get_employee
    print(hr_system.get_employee(1))  # {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}
    print(hr_system.get_employee(2))  # False

    # Test case for list_employees
    hr_system.employees = {1: {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}}
    print(hr_system.list_employees())  # {1: {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}}