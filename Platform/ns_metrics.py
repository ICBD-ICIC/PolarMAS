import os
import json
import statistics
from collections import Counter
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords

experiment = 'non_political'
#experiment = 'political'

# Folder containing JSON files
folder_path = f'outputs/cross_partisan_conversation/{experiment}'

# Create a list to hold all data
data = []

stop_words = set(stopwords.words('english'))

def analyze_discussions(df, label="All Parties"):
    print(f" ------------ {label} -----------------------\n")
    word_counter = Counter()
    words_per_message = []
    words_per_file = []
    messages_per_file = []

    for discussion in df['discussion']:
        messages_per_file.append(len(discussion))
        words_per_file_aux = 0
        for entry in discussion:
            words = tokenize(entry)
            words_per_file_aux += len(words)
            words_per_message.append(len(words))
            words = remove_stop_words(words)
            word_counter.update(words)
        words_per_file.append(words_per_file_aux)

    # Most common words
    most_common_words = word_counter.most_common(20)
    for word, count in most_common_words:
        print(f"{word}: {count}")

    # Medians
    if words_per_message:
        print(f"\nMedian number of words per message: {statistics.median(words_per_message)}")
    if words_per_file:
        print(f"Median number of words per run: {statistics.median(words_per_file)}")
    if messages_per_file:
        print(f"Median number of messages per run: {statistics.median(messages_per_file)}\n")

def polarization_metrics(df):
    print('Out group warmth change mean: ')
    print(df['out_group_warmth_change'].mean())

    print('Out group warmth change median: ')
    print(df['out_group_warmth_change'].median())

# Function to clean and tokenize text
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if 'agent' not in word]

def remove_stop_words(word_list):
    return [word for word in word_list if word not in stop_words]

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as f:
            content = json.load(f)

        agents_config = content.get('agents_config', '')
        number = int(agents_config.split('_')[-1].replace('.csv', ''))
        if number % 2 == 0:
            initiator = 'Democrat'
            democrat_name = 'Agent0'
        else:
            initiator = 'Republican'
            democrat_name = 'Agent1'
        pre = content.get('pre_questionnaire', {})
        post = content.get('post_questionnaire', {})
        discussion = content.get("discussion", [])[1:-1]

        # Extract identifier from filename or agents_config
        identifier = os.path.splitext(filename)[0]  # Without .json

        for agent in ['Agent0', 'Agent1']:
            # Extract pre and post Q1 and Q2
            def parse_qs(text):
                lines = text.strip().split('\n')
                return {line.split(':')[0].strip(): int(line.split(':')[1].strip()) for line in lines}


            pre_vals = parse_qs(pre.get(agent, 'Q1: 0\nQ2: 0'))
            post_vals = parse_qs(post.get(agent, 'Q1: 0\nQ2: 0'))

            data.append({
                'filename': filename,
                'discussion': discussion,
                'initiator': initiator,
                'agent': agent,
                'party': 'Democrat' if agent == democrat_name else 'Republican',
                'Q1_pre': pre_vals.get('Q1', None),
                'Q2_pre': pre_vals.get('Q2', None),
                'Q1_post': post_vals.get('Q1', None),
                'Q2_post': post_vals.get('Q2', None),
            })

# Convert to DataFrame
df = pd.DataFrame(data)
df['Q1_diff'] = df['Q1_post'] - df['Q1_pre']
df['Q2_diff'] = df['Q2_post'] - df['Q2_pre']
df['out_group_warmth_change'] = np.where(
    df['party'] == 'Republican',
    df['Q2_diff'],
    df['Q1_diff']
)

grouped = df.groupby('initiator')
for party, group in grouped:
    analyze_discussions(group, label=f"Initiator: {party}")

# --- Global analysis ---
analyze_discussions(df, label="All Parties Combined")

grouped = df.groupby('party')
for party, group in grouped:
    print(f" ------------ Metrics {party} -----------------------\n")
    polarization_metrics(group)

print(f" ------------ Metrics All Parties Combined -----------------------\n")
polarization_metrics(df)
