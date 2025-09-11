class HRManagementSystem:
    """
    This is a class for a personnel management system that implements functions such as adding, deleting, querying, and updating employees.
    """

    def __init__(self):
        """
        Initialize the HRManagementSystem with an attribute employees, which is an empty dictionary.
        """
        self.employees = {}

    def add_employee(self, employee_id, name, position, department, salary):
        """
        Add a new employee to the HRManagementSystem.
        :param employee_id: The employee's id, int.
        :param name: The employee's name, str.
        :param position: The employee's position, str.
        :param department: The employee's department, str.
        :param salary: The employee's salary, int.
        :return: If the employee is already in the HRManagementSystem, returns False, otherwise, returns True.
        """
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
        """
        Remove an employee from the HRManagementSystem.
        :param employee_id: The employee's id, int.
        :return: If the employee is already in the HRManagementSystem, returns True, otherwise, returns False.
        """
        if employee_id in self.employees:
            del self.employees[employee_id]
            return True
        return False

    def update_employee(self, employee_id, employee_info):
        """
        Update an employee's information in the HRManagementSystem.
        :param employee_id: The employee's id, int.
        :param employee_info: The employee's information, dict.
        :return: If the employee is already in the HRManagementSystem, returns True, otherwise, returns False.
        """
        if employee_id in self.employees:
            self.employees[employee_id].update(employee_info)
            return True
        return False

    def get_employee(self, employee_id):
        """
        Get an employee's information from the HRManagementSystem.
        :param employee_id: The employee's id, int.
        :return: If the employee is already in the HRManagementSystem, returns the employee's information, otherwise, returns False.
        """
        return self.employees.get(employee_id, False)

    def list_employees(self):
        """
        List all employees' information in the HRManagementSystem.
        :return: A list of all employees' information, dict.
        """
        return self.employees


if __name__ == "__main__":
    hr_management_system = HRManagementSystem()

    # Test case for add_employee
    print(hr_management_system.add_employee(1, 'John', 'Manager', 'Sales', 100000))  # True
    print(hr_management_system.add_employee(1, 'John', 'Manager', 'Sales', 100000))  # False

    # Test case for remove_employee
    print(hr_management_system.remove_employee(1))  # True
    print(hr_management_system.remove_employee(2))  # False

    # Adding back an employee for further tests
    hr_management_system.add_employee(1, 'John', 'Manager', 'Sales', 100000)

    # Test case for update_employee
    print(hr_management_system.update_employee(1, {'salary': 200000}))  # True
    print(hr_management_system.update_employee(2, {'salary': 200000}))  # False

    # Test case for get_employee
    print(hr_management_system.get_employee(1))  # {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 200000}
    print(hr_management_system.get_employee(2))  # False

    # Test case for list_employees
    print(hr_management_system.list_employees())  # {1: {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 200000}}