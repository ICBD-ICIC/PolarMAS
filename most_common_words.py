import os
import json
from collections import Counter
import re
from nltk.corpus import stopwords

experiment_type = 'h1_non_political'

# Path to the folder containing the JSON files
folder_path = f"outputs/{experiment_type}"
stop_words = set(stopwords.words('english'))

# Function to clean and tokenize text
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return  [word for word in words if word not in stop_words and 'agent' not in word]

# Counter for all words
word_counter = Counter()

# Iterate through all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            discussion = data.get("discussion", [])[1:-1]
            for entry in discussion:
                words = tokenize(entry)
                word_counter.update(words)

# Get the most common words
most_common_words = word_counter.most_common(20)

# Print the results
for word, count in most_common_words:
    print(f"{word}: {count}")
