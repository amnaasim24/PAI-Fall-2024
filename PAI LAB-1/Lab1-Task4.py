#Write a program that takes a list of numbers as input and returns the sum of all the elements in the list.

numbers = input("Enter a list of numbers separated by space: ")
total = 0
for char in numbers:
    if char != ' ':
        total += int(char)
print("The sum of the list is:", total)