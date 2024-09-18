# Programming for AI LAB - FALL 2024
# Lab - 5

from abc import ABC, abstractmethod
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width
    def area(self):
        return self.length * self.width
class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height
    def area(self):
        return 0.5 * self.base * self.height
class Square(Shape):
    def __init__(self, side):
        self.side = side
    def area(self):
        return self.side * self.side

r = Rectangle(4, 5)
t = Triangle(3, 4)
s = Square(6)

print("Rectangle area:", r.area())
print("Triangle area:", t.area())
print("Square area:", s.area())
