import os
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import spacy
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import normalize
from adjustText import adjust_text

import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ========== Global Settings ==========
TOP_N_WORDS = 20
STOP_WORDS = set(stopwords.words('english'))
NLP = spacy.load("en_core_web_md")  # Download: python -m spacy download en_core_web_md

# ========== Tokenization & Text Processing ==========

def tokenize(text):
    """Extract alphanumeric words excluding any containing 'agent'."""
    return [word for word in re.findall(r'\b\w+\b', text.lower()) if 'agent' not in word]

def remove_stop_words(words):
    return [word for word in words if word not in STOP_WORDS]

# ========== Analysis ==========

def analyze_discussions(df):
    """Aggregate word frequencies across all discussions in the DataFrame."""
    counter = Counter()
    for discussion in df['discussion']:
        for utterance in discussion:
            words = remove_stop_words(tokenize(utterance))
            counter.update(words)
    return counter

# ========== UMAP Visualization ==========

def generate_umap(word_counters):
    """Generate and save a UMAP visualization of word embeddings."""
    # Top N words per group
    top_words_by_group = {
        group: {word for word, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:TOP_N_WORDS]}
        for group, counter in word_counters.items()
    }

    # Determine which group(s) each word belongs to
    word_labels = {}
    word_sources = defaultdict(set)
    for group, words in top_words_by_group.items():
        for word in words:
            word_sources[word].add(group)

    for word, sources in word_sources.items():
        experiments_present = set(s.split('_')[1] for s in sources)
        parties_present = set(s.split('_')[0] for s in sources)

        if len(parties_present) > 1 and len(experiments_present) == 1:
            experiment_type = experiments_present.pop()
            word_labels[word] = 'Both_' + experiment_type
        else:
            word_labels[word] = list(sources)[0]

    # Filter words with vectors
    filtered_words = []
    filtered_labels = []
    for word, label in sorted(word_labels.items()):
        token = NLP.vocab[word]
        if token.has_vector:
            filtered_words.append(word)
            filtered_labels.append(label)

    vectors = normalize(np.array([NLP.vocab[word].vector for word in filtered_words]))

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(vectors)

    # Plotting
    color_map = {
        'Democrat_political': '#0000ff',
        'Republican_political': '#ff0000',
        'Democrat_non_political': '#00ccff',
        'Republican_non_political': '#ff00cc',
        'Both_political': '#8800ff',
        'Both_non_political': '#cc00ff'
    }
    marker_map = {
        'Democrat_political': 'o',
        'Republican_political': 's',
        'Democrat_non_political': '^',
        'Republican_non_political': 'D',
        'Both_political': 'X',
        'Both_non_political': 'P'
    }

    plt.figure(figsize=(14, 10))
    for label in sorted(set(filtered_labels)):
        if label.startswith('Both_'):
            continue  # Skip words shared by multiple groups

        indices = [i for i, l in enumerate(filtered_labels) if l == label]
        color = color_map.get(label, 'gray')
        marker = marker_map.get(label, 'o')

        plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            s=150,  # Increased marker size
            edgecolor='k',
            alpha=0.8,
            color=color,
            marker=marker,
            label=label.replace('_', ' ')
        )

        # Use adjustText to prevent overlap
        texts = []
        for j in indices:
            x, y = embedding[j]
            label = filtered_words[j]
            group_label = filtered_labels[j]
            print(f"{filtered_words[j]} -> {group_label}")
            texts.append(plt.text(x, y, label, fontsize=15))
        adjust_text(
            texts,
            expand_text=(2, 2),
            expand_points=(3, 3),
            force_text=1,
            force_points=1,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
            autoalign='y',
            only_move={'points': 'xy', 'texts': 'xy'},
            lim=1000
        )

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    with PdfPages("umap_word_visualization.pdf") as pdf:
        pdf.savefig()
        plt.close()


# ========== Data Ingestion ==========

def load_data(experiments_dir):
    data = []
    for experiment in ['non_political', 'political']:
        folder_path = os.path.join(experiments_dir, experiment)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
                    continue

                agents_config = content.get('agents_config', '')
                try:
                    number = int(agents_config.split('_')[-1].replace('.csv', ''))
                except Exception:
                    continue

                initiator = 'Democrat' if number % 2 == 0 else 'Republican'
                democrat_name = 'Agent0' if initiator == 'Democrat' else 'Agent1'
                discussion = content.get("discussion", [])[1:-1]  # Remove initial instructions and summary

                for agent in ['Agent0', 'Agent1']:
                    data.append({
                        'discussion': discussion,
                        'initiator': initiator,
                        'party': 'Democrat' if agent == democrat_name else 'Republican',
                        'experiment': experiment
                    })
    return pd.DataFrame(data)

# ========== Main Pipeline ==========

def main():
    df = load_data('outputs/cross_partisan_conversation')
    if df.empty:
        print("No data found.")
        return

    word_counters = defaultdict(Counter)
    for (initiator, experiment), group in df.groupby(['initiator', 'experiment']):
        key = f"{initiator}_{experiment}"
        word_counters[key] = analyze_discussions(group)

    generate_umap(word_counters)

if __name__ == "__main__":
    main()