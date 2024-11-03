
def count_characters_and_words(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            char_count = len(content)
            word_count = len(content.split())
            print(f"Total number of characters: {char_count}")
            print(f"Total number of words: {word_count}")
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

filename = 'sample.txt'
count_characters_and_words(filename)
