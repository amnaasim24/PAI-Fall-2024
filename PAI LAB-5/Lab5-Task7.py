# Programming for AI LAB - FALL 2024
# Lab - 5

def extract_emails(text):
    emails = []
    words = text.split()

    for word in words:
        if "@" in word and "." in word:
            at_index = word.index('@')
            dot_index = word.rindex('.')

            if at_index > 0 and dot_index > at_index + 1 and dot_index < len(word) - 1:
                emails.append(word.strip(",.;:"))

    return emails


text = """Please contact us at support@example.com or sales@company.org. You can also reach out to admin@domain.co.in."""
extracted_emails = extract_emails(text)

print(extracted_emails)
