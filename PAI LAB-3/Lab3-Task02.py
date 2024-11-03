def search_and_replace(filename, search_text, replace_text):
    try:
        with open(filename, 'r') as file:
            content = file.read()

        new_content = content.replace(search_text, replace_text)

        with open(filename, 'w') as file:
            file.write(new_content)

        print("Replacement complete.")
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


filename = 'sample.txt'
search_text = 'VS Code'
replace_text = 'Pycharm'
search_and_replace(filename, search_text, replace_text)
