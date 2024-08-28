def UserFeedback():
    name=input("Your good name: ")
    print("Hello ", name, "!")
    print("I hope you are fine, let me know how I can help you!")

    response=input("Do you need any help? (yes/no): ").strip().lower()
    if response=='yes':
        problem=input("Share your problem with us: ")
        print("Thanks for your feedback, we will resolve your problem!")
    else:
        print('Okay! Good to see you , stay connected.'.center(40))

UserFeedback()