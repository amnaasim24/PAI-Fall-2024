class Vehicle:
    def __init__(self, seating_capacity):
        self.seating_capacity = seating_capacity

    def calculate_fare(self):
        """Calculate the base fare for the vehicle."""
        return self.seating_capacity * 100


class Bus(Vehicle):
    def __init__(self, seating_capacity):
        super().__init__(seating_capacity)

    def calculate_fare(self):
        """Calculate the fare for the bus including a 10% maintenance charge."""
        base_fare = super().calculate_fare()
        maintenance_charge = 0.10 * base_fare
        total_fare = base_fare + maintenance_charge
        return total_fare


if __name__ == "__main__":
    vehicle = Vehicle(seating_capacity=30)
    print(f"Vehicle fare: ${vehicle.calculate_fare():.2f}")

    bus = Bus(seating_capacity=30)
    print(f"Bus fare: ${bus.calculate_fare():.2f}")
