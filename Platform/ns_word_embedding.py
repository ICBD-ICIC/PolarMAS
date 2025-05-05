import os
import json
import statistics
from collections import Counter
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import umap
import spacy

from sklearn.preprocessing import normalize

# Load English tokenizer, POS tagger, etc.
nlp = spacy.load("en_core_web_md")  # Make sure to run: python -m spacy download en_core_web_md

data = []
stop_words = set(stopwords.words('english'))

def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if 'agent' not in word]

def remove_stop_words(word_list):
    return [word for word in word_list if word not in stop_words]

def analyze_discussions(df, label="All Parties", return_counter=False):
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

    most_common_words = word_counter.most_common(1000)  # Top 100 for UMAP
    # for word, count in most_common_words[:20]:
    #     print(f"{word}: {count}")

    # if words_per_message:
    #     print(f"\nMedian number of words per message: {statistics.median(words_per_message)}")
    # if words_per_file:
    #     print(f"Median number of words per run: {statistics.median(words_per_file)}")
    # if messages_per_file:
    #     print(f"Median number of messages per run: {statistics.median(messages_per_file)}\n")

    if return_counter:
        return word_counter
    else:
        return dict(most_common_words)

def generate_umap(word_counters_by_group):
    from matplotlib.backends.backend_pdf import PdfPages
    from collections import defaultdict

    # Sort top words deterministically
    top_words_by_group = {
        group: set([
            word for word, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:30]
        ])
        for group, counter in word_counters_by_group.items()
    }

    # Make word order deterministic
    all_words = sorted(set().union(*top_words_by_group.values()))

    word_labels = {}
    word_sources = defaultdict(set)

    for group, words in top_words_by_group.items():
        for word in words:
            word_sources[word].add(group)

    for word, groups in word_sources.items():
        if len(groups) > 1:
            word_labels[word] = 'Both'
        else:
            word_labels[word] = list(groups)[0]

    filtered_words = []
    filtered_labels = []

    for word, label in sorted(word_labels.items()):  # deterministic order
        token = nlp.vocab[word]
        if token.has_vector:
            filtered_words.append(word)
            filtered_labels.append(label)

    vectors = [nlp.vocab[word].vector for word in filtered_words]
    vectors = normalize(np.array(vectors))

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(vectors)

    color_map = {
        'Democratpolitical': '#0000ff',
        'Republicanpolitical': '#ff0000',
        'Democratnonpolitical': '#00ccff',
        'Republicannonpolitical': '#ff00cc',
        'Both': '#8800ff'
    }

    marker_map = {
        'Democratpolitical': 'o',
        'Republicanpolitical': 's',
        'Democratnonpolitical': '^',
        'Republicannonpolitical': 'D',
        'Both': 'X'
    }

    plt.figure(figsize=(14, 10))
    for label in sorted(set(filtered_labels)):  # sorted for consistent legend order
        indices = [i for i, l in enumerate(filtered_labels) if l == label]
        if not indices:
            continue

        label_clean = label.replace('_', ' ')
        color = color_map[label_clean]
        marker = marker_map[label_clean]

        plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            s=50,
            edgecolor='k',
            alpha=0.8,
            color=color,
            marker=marker,
            label=label_clean
        )
        for j in indices:
            plt.text(embedding[j, 0] + 0.01, embedding[j, 1] + 0.01, filtered_words[j], fontsize=12)

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to PDF
    with PdfPages("umap_word_visualization.pdf") as pdf:
        pdf.savefig()
        plt.close()

experiments = ['non_political', 'political']
data = []

for experiment in experiments:
    folder_path = f'outputs/cross_partisan_conversation/{experiment}'

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
            discussion = content.get("discussion", [])[1:-1]
            identifier = os.path.splitext(filename)[0]

            for agent in ['Agent0', 'Agent1']:
                def parse_qs(text):
                    lines = text.strip().split('\n')
                    return {line.split(':')[0].strip(): int(line.split(':')[1].strip()) for line in lines}

                data.append({
                    'discussion': discussion,
                    'initiator': initiator,
                    'party': 'Democrat' if agent == democrat_name else 'Republican',
                    'experiment': experiment
                })


df = pd.DataFrame(data)

from collections import defaultdict

word_counters = defaultdict(Counter)
grouped = df.groupby(['initiator', 'experiment'])

for (party, experiment_type), group in grouped:
    counter = analyze_discussions(group, label=f"{party} - {experiment_type}", return_counter=True)
    key = f"{party}_{experiment_type}"
    word_counters[key] = counter

# Analyze discussions and get common words for UMAP
#all_words = analyze_discussions(df, label="All Parties Combined")
generate_umap(word_counters)
