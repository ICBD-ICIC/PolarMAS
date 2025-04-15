import os
import json
import statistics
from collections import Counter
import re

import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

experiment_type = 'h1_political'

# Path to the folder containing the JSON files
folder_path = f"outputs/{experiment_type}"
stop_words = set(stopwords.words('english'))


# Function to clean and tokenize text
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if 'agent' not in word]


def remove_stop_words(word_list):
    return [word for word in word_list if word not in stop_words]


# Counter for all words
word_counter = Counter()

words_per_message = []
words_per_file = []
messages_per_file = []

# Iterate through all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            discussion = data.get("discussion", [])[1:-1]
            messages_per_file.append(len(discussion))
            words_per_file_aux = 0
            for entry in discussion:
                words = tokenize(entry)
                words_per_file_aux += len(words)
                words_per_message.append(len(words))
                words = remove_stop_words(words)
                word_counter.update(words)
            words_per_file.append(words_per_file_aux)

# Get the most common words
most_common_words = word_counter.most_common(20)

# Print the results
for word, count in most_common_words:
    print(f"{word}: {count}")

# Calculate and print medians
if words_per_message:
    median_words = statistics.median(words_per_message)
    print(f"\nMedian number of words per message: {median_words}")

if words_per_message:
    median_words = statistics.median(words_per_file)
    print(f"Median number of words per run: {median_words}")

if messages_per_file:
    median_messages = statistics.median(messages_per_file)
    print(f"Median number of messages per run: {median_messages}")
