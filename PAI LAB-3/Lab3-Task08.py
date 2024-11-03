
try:
    num1 = int(input("Enter first Number: "))
    num2 = int(input("Enter second Number: "))

    result = num1 / num2
    print(f"The result is {result}")

except ZeroDivisionError:
    print("Error: Number cannot be divided by 0")
except ValueError:
    print("Error: Incorrect datatype")
