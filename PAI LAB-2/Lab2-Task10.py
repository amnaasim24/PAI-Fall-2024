
def build_message(**info):
    message_parts = []
    for key, value in info.items():
        message_parts.append(f"{key.capitalize()}: {value}")

    summary_message = "\n".join(message_parts)

    return summary_message

message = build_message(name=  "Huzaifa", age = 19, city = "New York", hobby = "painting")
print(message)
