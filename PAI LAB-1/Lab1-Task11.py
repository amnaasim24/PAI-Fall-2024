marks_dict = {}
marks_dict['subject1'] = float(input("Enter marks for Subject 1: "))
marks_dict['subject2'] = float(input("Enter marks for Subject 2: "))
marks_dict['subject3'] = float(input("Enter marks for Subject 3: "))

total_marks = marks_dict['subject1'] + marks_dict['subject2'] + marks_dict['subject3']

average_marks = total_marks / 3
percentage = (total_marks / 300) * 100

print(f"Total Marks: {total_marks}")
print(f"Average Marks: {average_marks:.2f}")
print(f"Percentage: {percentage:.2f}%")
