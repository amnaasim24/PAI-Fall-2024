import json

dictionary = {'Ali': 23, 'Saad': 24, 'Salman': 15, 'Shams': 25, 'Sadiq': 46, 'Hammad': 23}

try:
    with open('data.json', 'w') as json_file:
        json.dump(dictionary, json_file)
    print("Dictionary saved to JSON file.")
except Exception as e:
    print(f"Error while saving dictionary: {e}")

try:
    with open('data.json', 'r') as json_file:
        loaded_data = json.load(json_file)
    print("Dictionary loaded from JSON file.")
except Exception as e:
    print(f"Error while loading dictionary: {e}")

try:
    max_age = max(loaded_data.values())
    max_age_person = [name for name, age in loaded_data.items() if age == max_age][0]
    print(f"The person with the maximum age is {max_age_person} with an age of {max_age}.")

    seen_ages = {}

    for name, age in loaded_data.items():
        if age in seen_ages:
            seen_ages[age].append(name)
        else:
            seen_ages[age] = [name]

    for age, people in seen_ages.items():
        if len(people) > 1:
            print(f"People with age {age}: {', '.join(people)}")

except ValueError:
    print("Error: The dictionary is empty.")
except Exception as e:
    print(f"An error occurred: {e}")
