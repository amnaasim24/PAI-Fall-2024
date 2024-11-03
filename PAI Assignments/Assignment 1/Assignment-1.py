class Animal():
    def __init__(self, name, age, hab):
        self.name = name
        self.age = age
        self.hab = hab
        self.is_available = True

    def set_availability(self, is_available):
        self.is_available = is_available

    def display_info(self):
        pass

class Mammal(Animal):
    def __init__(self, name, age, hab, fur, diet):
        super().__init__(name, age, hab)
        self.fur = fur
        self.diet = diet

    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Habitat: {self.hab}")
        print(f"Fur Length: {self.fur}")
        print(f"Diet Type: {self.diet}")
        print(f"Available: {self.is_available}")

class Bird(Animal):
    def __init__(self, name, age, hab, wings, flight):
        super().__init__(name, age, hab)
        self.wings = wings
        self.flight = flight

    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Habitat: {self.hab}")
        print(f"Wingspan: {self.wings}")
        print(f"Flight Altitude: {self.flight}")
        print(f"Available : {self.is_available}")

class Reptile(Animal):
    def __init__(self, name, age, hab, pattern, venom):
        super().__init__(name, age, hab)
        self.pattern = pattern
        self.venom = venom

    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Habitat: {self.hab}")
        print(f"Scale Pattern: {self.pattern}")
        print(f"Venomous: {self.venom}")
        print(f"Available : {self.is_available}")


lion = Mammal("Lion", 8, "Savanna", "Long", "Carnivore")
eagle = Bird("Eagle", 5, "Mountains", "2.5 meters", "1000 meters")
python = Reptile("Python", 6, "Jungle", "Smooth", False)

print("Mammal:")
lion.display_info()
print("\nBird:")
eagle.display_info()
print("\nReptile:")
python.display_info()

print("\nQuarantining the Lion...")
lion.set_availability(False)
print("Mammal:")
lion.display_info()