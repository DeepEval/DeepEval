class HRManagementSystem:
    """
    This is a class as personnel management system that implements functions such as adding, deleting, querying, and updating employees
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
        employee_list = []
        for employee_id, info in self.employees.items():
            employee_list.append({
                'employee_ID': employee_id,
                'name': info['name'],
                'position': info['position'],
                'department': info['department'],
                'salary': info['salary']
            })
        return employee_list


if __name__ == "__main__":
    hrManagementSystem = HRManagementSystem()
    
    # Test add_employee
    print(hrManagementSystem.add_employee(1, 'John', 'Manager', 'Sales', 100000))  # True
    print(hrManagementSystem.add_employee(1, 'John', 'Manager', 'Sales', 100000))  # False
    
    # Test remove_employee
    print(hrManagementSystem.remove_employee(1))  # True
    print(hrManagementSystem.remove_employee(2))  # False
    
    # Test update_employee
    hrManagementSystem.employees = {1: {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}}
    print(hrManagementSystem.update_employee(1, {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 20000}))  # True
    print(hrManagementSystem.update_employee(2, {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 20000}))  # False
    
    # Test get_employee
    hrManagementSystem.employees = {1: {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}}
    print(hrManagementSystem.get_employee(1))  # {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}
    print(hrManagementSystem.get_employee(2))  # False
    
    # Test list_employees
    hrManagementSystem.employees = {1: {'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}}
    print(hrManagementSystem.list_employees())  # [{'employee_ID': 1, 'name': 'John', 'position': 'Manager', 'department': 'Sales', 'salary': 100000}]