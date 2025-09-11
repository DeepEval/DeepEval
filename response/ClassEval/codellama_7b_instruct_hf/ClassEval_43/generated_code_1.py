import collections

class HRManagementSystem:
    def __init__(self):
        self.employees = collections.defaultdict(dict)

    def add_employee(self, employee_id, name, position, department, salary):
        if employee_id in self.employees:
            return False
        self.employees[employee_id] = {
            'name': name,
            'position': position,
            'department': department,
            'salary': salary
        }
        return True

    def remove_employee(self, employee_id):
        if employee_id not in self.employees:
            return False
        del self.employees[employee_id]
        return True

    def update_employee(self, employee_id, employee_info):
        if employee_id not in self.employees:
            return False
        self.employees[employee_id].update(employee_info)
        return True

    def get_employee(self, employee_id):
        if employee_id not in self.employees:
            return False
        return self.employees[employee_id]

    def list_employees(self):
        return self.employees.copy()

if __name__ == "__main__":
    hrManagementSystem = HRManagementSystem()
    output = hrManagementSystem.add_employee(1, 'John', 'Manager', 'Sales', 100000)
    print(output)
    output = hrManagementSystem.add_employee(2, 'Jane', 'Manager', 'Sales', 120000)
    print(output)
    output = hrManagementSystem.remove_employee(1)
    print(output)
    output = hrManagementSystem.update_employee(2, {'name': 'Jane', 'salary': 150000})
    print(output)
    output = hrManagementSystem.get_employee(2)
    print(output)
    output = hrManagementSystem.list_employees()
    print(output)