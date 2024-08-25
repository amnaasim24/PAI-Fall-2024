#Write a program to take a list and a number input from user and then delete all elements in the list less than that number.

numbers = input("Enter a list of numbers separated by space: ")

num_list = []
num_str = ""
for char in numbers:
    if char == " ":
        num_list += [int(num_str)]
        num_str = ""
    else:
        num_str += char
num_list += [int(num_str)]

num = int(input("Enter a number: "))

i = 0
while i < len(num_list):
    if num_list[i] < num:
        num_list = num_list[:i] + num_list[i+1:]
    else:
        i += 1

print("The list after deleting elements less than", num, "is:", num_list)
