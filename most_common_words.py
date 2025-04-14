import os
import json
from collections import Counter
import re
from nltk.corpus import stopwords

# Path to the folder containing the JSON files
folder_path = "outputs/h1_non_political_reverse"
stop_words = set(stopwords.words('english'))

# Intro message to remove
intro_message = (
    "\nWe've randomly assigned you a partner that belongs to or leans toward your out party. "
    "Please have a conversation by chatting with them about the meaning of life.\nSpecifically, we are interested in you sharing "
    "what you think makes life meaningful and learning your conversation partner's thoughts as someone that might hold different values and beliefs.\n"
    "For example, survey research shows that many people mention family as the most important sources of meaning in their life. Survey research also shows "
    "that other people mention career, money, faith, friends, and hobbies as the most important source of meaning in their life. \n"
    "What do you think?. Limit your words to 50.\nEverytime you respond, respond with <name>:."
)

# Function to clean and tokenize text
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return  [word for word in words if word not in stop_words]

# Counter for all words
word_counter = Counter()

# Iterate through all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            discussion = data.get("discussion", [])
            for entry in discussion:
                if entry.strip() == intro_message.strip():
                    continue  # Skip the intro message
                words = tokenize(entry)
                word_counter.update(words)

# Get the most common words
most_common_words = word_counter.most_common(50)  # Top 50

# Print the results
for word, count in most_common_words:
    print(f"{word}: {count}")
