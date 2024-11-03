def save_employee_data(filename):
    try:
        name = input("Enter employee name: ")
        cnic = input("Enter CNIC number: ")
        age = input("Enter age: ")
        salary = input("Enter salary: ")

        with open(filename, 'w') as file:
            file.write(f"Name: {name}\nCNIC: {cnic}\nAge: {age}\nSalary: {salary}\n")

        contact_number = input("Enter contact number: ")

        with open(filename, 'a') as file:
            file.write(f"Contact Number: {contact_number}\n")

        with open(filename, 'r') as file:
            content = file.read()
            print("File content:")
            print(content)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


filename = 'employee_biodata.txt'
save_employee_data(filename)
