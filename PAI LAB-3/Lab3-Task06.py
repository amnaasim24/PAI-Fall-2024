def write_question_to_file():
    try:
        sentence = input("Please enter a sentence: ").strip()

        if sentence.endswith('?'):
            with open('questions.txt', 'a') as file:
                file.write(sentence + '\n')
            print("The sentence has been written to questions.txt.")
        else:
            print("The sentence is not a question and was not saved.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


write_question_to_file()
