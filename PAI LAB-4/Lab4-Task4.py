# Programming for AI LAB - FALL 2024
# lab - 4

class Employee:
    def __init__(self):
        self.name = ""
        self.monthly_salary = 0
        self.tax_percentage = 0.02
    def get_data(self):
        self.name = input("Enter employee name: ")
        self.monthly_salary = float(input("Enter monthly salary: "))
    def salary_after_tax(self):
        return self.monthly_salary * (1 - self.tax_percentage)
    def update_tax_percentage(self, new_tax_percentage):
        self.tax_percentage = new_tax_percentage / 100

emp = Employee()
emp.get_data()

print("Salary after tax:", emp.salary_after_tax())
emp.update_tax_percentage(3)
print("Salary after tax (updated):", emp.salary_after_tax())
