# Programming for AI LAB - FALL 2024
# Lab - 5

class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def calculateBonus(self):
        pass

    def display_employee_details(self):
        print(f"Employee: {self.name}, Salary: {self.salary}")

class Manager(Employee):
    def calculateBonus(self):
        return self.salary * 0.20

    def hire(self):
        print(f"{self.name} is hiring someone.")

class Developer(Employee):
    def calculateBonus(self):
        return self.salary * 0.10

    def writeCode(self):
        print(f"{self.name} is writing code.")

class SeniorManager(Manager):
    def calculateBonus(self):
        return self.salary * 0.30


manager = Manager("Alice", 100000)
developer = Developer("Bob", 80000)
senior_manager = SeniorManager("Charlie", 150000)

manager.display_employee_details()
print(f"Manager Bonus: {manager.calculateBonus()}")
manager.hire()

developer.display_employee_details()
print(f"Developer Bonus: {developer.calculateBonus()}")
developer.writeCode()

senior_manager.display_employee_details()
print(f"Senior Manager Bonus: {senior_manager.calculateBonus()}")
senior_manager.hire()
