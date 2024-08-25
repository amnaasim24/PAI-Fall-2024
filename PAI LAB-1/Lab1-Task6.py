physics_marks = int(input("Enter Aliza's Physics marks: "))
chemistry_marks = int(input("Enter Aliza's Chemistry marks: "))
maths_marks = int(input("Enter Aliza's Maths marks: "))

marks_dict = {"Physics": physics_marks, "Chemistry": chemistry_marks, "Maths": maths_marks}

total_marks = physics_marks + chemistry_marks + maths_marks
average_marks = total_marks / 3

if physics_marks > chemistry_marks:
    if physics_marks > maths_marks:
        highest_marks_subject = "Physics"
    else:
        highest_marks_subject = "Maths"
else:
    if chemistry_marks > maths_marks:
        highest_marks_subject = "Chemistry"
    else:
        highest_marks_subject = "Maths"

print("\nAliza's average marks are:", average_marks)
print("Aliza got the highest marks in", highest_marks_subject)