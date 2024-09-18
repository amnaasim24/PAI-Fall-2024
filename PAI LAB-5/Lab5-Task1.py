# Programming for AI LAB - FALL 2024
# Lab - 5

class Vehicle:
    def __init__(self, seating_capacity):
        self.seating_capacity = seating_capacity
    def calculate_fare(self):
        return self.seating_capacity * 100
class Bus(Vehicle):
    def __init__(self, seating_capacity):
        super().__init__(seating_capacity)
    def calculate_fare(self):
        base_fare = super().calculate_fare()
        maintenance_charge = base_fare * 0.10
        return base_fare + maintenance_charge

b = Bus(50)
print(b.calculate_fare())
