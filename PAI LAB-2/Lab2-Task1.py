def trapezoid_area():
    print("Data for calculating Area of Trapezoid:")
    b1= int(input("Enter the value of Base1: "))
    b2= int(input("Enter the value of Base2: "))
    h= int(input("Enter the value of Height: "))
    print("\nThe calculated Area of Trapezoid is: ", (((b1+b2)/2)*h))

def parallelogram_area():
    print("\nData for calculating Area of Parallelogram:")
    b = int(input("Enter the value of Base: "))
    h = int(input("Enter the value of Height: "))
    print("\nThe calculated Area of Parallelogram is: ", b*h)

def cylinder_surface_area():
    print("\nData for calculating Surface Area of Cylinder:")
    r = int(input("Enter the value of Radius: "))
    h = int(input("Enter the value of Height: "))
    pi = 3.14159
    print("\nThe calculated Surface Area of Cylinder is: ", (2*pi*r*(r+h)))

def cylinder_volume():
    print("\nData for calculating Volume of Cylinder:")
    r = int(input("Enter the value of Radius: "))
    h = int(input("Enter the value of Height: "))
    pi = 3.14159
    print("\nThe calculated Volume of Cylinder is: ", (pi*r*r*h) )

trapezoid_area()
parallelogram_area()
cylinder_surface_area()
cylinder_volume()
