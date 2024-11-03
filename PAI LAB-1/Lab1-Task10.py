numbers = int(input("Enter the number of elements in the list: "))
numbers_list = []
for i in range (numbers):
    number = float(input(f"Enter element {i + 1}: "))
    numbers_list.append(number)

largest_number = max(numbers_list)
print("The largest number in the list is:", largest_number)
