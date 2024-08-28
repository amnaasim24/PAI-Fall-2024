def find_maximum(*numbers):
    max=0
    for i in numbers:
        if i>max:
            max=i
    print("The Maximum number is: ", max)

find_maximum(1,2,3,4,5,6,7,8,9)
