Score = int(input("Enter your Score:"))

if Score>=80:
    if Score>85:
        print("CS Section: A")
    else:
        print("CS Section: B")
elif Score>=70:
    if Score>75:
        print("AI Section: A")
    else:
        print("AI Section: B")
elif Score>=60:
    if Score>65:
        print("DS Section: A")
    else:
        print("DS Section: B")
