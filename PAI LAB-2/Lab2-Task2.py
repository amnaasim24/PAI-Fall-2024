def checklastletter(givenstring):
    vowels = 'aeiouAEIOU'
    lastletter = givenstring[-1]
    if lastletter in vowels:
        print("The last letter is a vowel.")
    else:
        print("The last character is a consonant.")

givenstring = input("Enter a string: ")
print(checklastletter(givenstring))
