
def lists_to_dict(list1, list2):
    result_dict = {}
    for i in range(len(list1)):
        result_dict[list1[i]] = list2[i]

    return result_dict

list1 = input("Enter the first list of elements separated by spaces: ").split()
list2 = input("Enter the second list of elements separated by spaces: ").split()

if len(list1) != len(list2):
    print("Error: Both lists must have the same number of elements.")
else:
    result = lists_to_dict(list1, list2)
    print("The resulting dictionary is:", result)
