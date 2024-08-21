num_list = input("Enter a list of numbers separated by space: ")

num_list = num_list.split()
num_list = [int(x) for x in num_list]

even_count = 0

for num in num_list:
    if num % 2 == 0:
        even_count += 1

print("Count of even numbers:", even_count)