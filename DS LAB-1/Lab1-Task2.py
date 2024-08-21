a = int(input("Enter the first Integer: "))
b = int(input("Enter the second Integer: "))
c = int(input("Choose the Operation you want to perform:\n1. + \n2. - \n3. * \n4. / \n"))

if c==1:
    print(a--b)
elif c==2:
    print(a-b)
elif c==3:
    print(a*b)
elif c==4:
    print(a/b)
