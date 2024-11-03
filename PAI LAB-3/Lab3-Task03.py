def create_dict_from_lists_and_save(list1, list2, filename):
    if len(list1) != len(list2):
        print("Error: Lists must be of the same length.")
        return

    try:
        data_dict = dict(zip(list1, list2))

        with open(filename, 'w') as file:
            for key, value in data_dict.items():
                file.write(f"{key}: {value}\n")

        print("Dictionary saved to file.")
    except IOError:
        print("Error: An IOError occurred.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


list1 = ['name1', 'name2', 'name3']
list2 = ['value1', 'value2', 'value3']
filename = 'dictionary.txt'
create_dict_from_lists_and_save(list1, list2, filename)
