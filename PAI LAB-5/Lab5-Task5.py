# Programming for AI LAB - FALL 2024
# Lab - 5

from datetime import date

class Vehicle:
    def __init__(self, make, model, rental_price_per_day):
        self._make = make
        self._model = model
        self._rental_price_per_day = rental_price_per_day
        self._is_available = True

    def is_available(self):
        return self._is_available

    def rent(self):
        if self._is_available:
            self._is_available = False
            return True
        return False

    def return_vehicle(self):
        self._is_available = True

    def calculate_rental_cost(self, rental_days):
        return self._rental_price_per_day * rental_days

    def display_details(self):
        print(f"Make: {self._make}, Model: {self._model}, Price per day: ${self._rental_price_per_day}, Available: {self._is_available}")

class Car(Vehicle):
    def __init__(self, make, model, rental_price_per_day):
        super().__init__(make, model, rental_price_per_day)

class SUV(Vehicle):
    def __init__(self, make, model, rental_price_per_day):
        super().__init__(make, model, rental_price_per_day)

class Truck(Vehicle):
    def __init__(self, make, model, rental_price_per_day):
        super().__init__(make, model, rental_price_per_day)

class Customer:
    def __init__(self, name, contact_info):
        self._name = name
        self._contact_info = contact_info
        self._rental_history = []

    def add_rental(self, rental):
        self._rental_history.append(rental)

    def display_rental_history(self):
        print(f"Rental history for {self._name}:")
        for rental in self._rental_history:
            rental.display_reservation_details()

class RentalReservation:
    def __init__(self, customer, vehicle, start_date, end_date):
        self.customer = customer
        self.vehicle = vehicle
        self.start_date = start_date
        self.end_date = end_date
        self.rental_days = (end_date - start_date).days
        self.total_cost = self.vehicle.calculate_rental_cost(self.rental_days)

    def display_reservation_details(self):
        print(f"Customer: {self.customer._name}")
        self.vehicle.display_details()
        print(f"Rental Start Date: {self.start_date}")
        print(f"Rental End Date: {self.end_date}")
        print(f"Total Rental Days: {self.rental_days}")
        print(f"Total Cost: ${self.total_cost}")

def display_vehicle_or_reservation(item):
    item.display_details()
car = Car("Toyota", "Corolla", 40)
suv = SUV("Ford", "Explorer", 60)
truck = Truck("Ram", "1500", 75)

customer = Customer("Huzaifa", "shuzaifaali05@gmail.com")

rental_start = date(2024, 9, 1)
rental_end = date(2024, 9, 5)
reservation = RentalReservation(customer, car, rental_start, rental_end)

customer.add_rental(reservation)

if car.rent():
    print("Vehicle has been rented.")
else:
    print("Vehicle is not available.")

customer.display_rental_history()

car.return_vehicle()

display_vehicle_or_reservation(car)
display_vehicle_or_reservation(suv)
