
students_grades = {}

def add_grade(student_name, grade):
    if student_name in students_grades:
        students_grades[student_name].append(grade)
    else:
        print(f"Student {student_name} not found.")

def calculate_average(student_name):
    if student_name in students_grades:
        grades = students_grades[student_name]
        if grades:
            average = sum(grades) / len(grades)
            print(f"{student_name}'s average grade is {average:.2f}")
        else:
            print(f"{student_name} has no grades yet.")
    else:
        print(f"Student {student_name} not found.")

def add_student(student_name):
    if student_name not in students_grades:
        students_grades[student_name] = []
        print(f"Student {student_name} added.")
    else:
        print(f"Student {student_name} already exists.")

def remove_student(student_name):
    if student_name in students_grades:
        del students_grades[student_name]
        print(f"Student {student_name} removed.")
    else:
        print(f"Student {student_name} not found.")

add_student("Sabeeh")
add_student("Haris")

add_grade("Sabeeh", 90)
add_grade("Sabeeh", 85)
add_grade("Haris", 78)

calculate_average("Sabeeh")
calculate_average("Haris")

remove_student("Haris")
