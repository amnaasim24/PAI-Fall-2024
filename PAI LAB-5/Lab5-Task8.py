# Programming for AI LAB - FALL 2024
# Lab - 5

import re

text = """His birthday is on 12/09/2023. The project deadline is 2023-09-12, and we also have a meeting scheduled for Sep 12, 2023. Another date is 09/12/2023.
"""

# 1. MM/DD/YYYY or DD/MM/YYYY
pattern1 = r'\b\d{2}/\d{2}/\d{4}\b'

# 2. YYYY-MM-DD
pattern2 = r'\b\d{4}-\d{2}-\d{2}\b'

# 3. Month DD, YYYY (e.g., Sep 12, 2023)
pattern3 = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},\s\d{4}\b'

# Combine all patterns using the OR '|' operator
combined_pattern = f'({pattern1})|({pattern2})|({pattern3})'

matches = re.findall(combined_pattern, text)

extracted_dates = [match[0] or match[1] or match[2] for match in matches]

print("Extracted Dates:", extracted_dates)
