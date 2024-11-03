def replace_incorrect_letter(file_name):
    try:
        with open(file_name, 'r') as file:
            content = file.read()
            print(content)

        incorrect_letter = 'q'
        correct_letter = 's'
        if incorrect_letter in content:
            corrected_content = content.replace(incorrect_letter, correct_letter)
            print("Letter replacement successful")
        else:
            print(f"No {incorrect_letter} found")

        print("Corrected sentence:")
        print(corrected_content)

    except FileNotFoundError:
        print("Error: File not found")


replace_incorrect_letter("replacement_needed.txt")
