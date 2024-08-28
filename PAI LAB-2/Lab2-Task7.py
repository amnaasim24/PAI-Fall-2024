temperatures = [23, 25, 28, 22, 20, 24, 26, 27, 29, 30, 25, 24, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]

average_temp = sum(temperatures) / len(temperatures)
print("Average temperature:", average_temp)

print("Highest temperature:", max(temperatures))
print("Lowest temperature:", min(temperatures))

temperatures.sort()
print("Temperatures in order:", temperatures)

day_to_remove = 10
temperatures.pop(day_to_remove - 1)
print("Temperatures after removing day", day_to_remove, ":", temperatures)
