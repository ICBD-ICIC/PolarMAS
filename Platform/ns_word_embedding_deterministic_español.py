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

import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ========== Configuración global ==========
TOP_N_WORDS = 20  # Número de palabras más frecuentes por grupo
STOP_WORDS = set(stopwords.words('english'))
NLP = spacy.load("en_core_web_md")  # Descargar con: python -m spacy download en_core_web_md

# ========== Tokenización y procesamiento de texto ==========

def tokenize(text):
    """Extrae palabras alfanuméricas, excluyendo las que contienen 'agent'."""
    return [word for word in re.findall(r'\b\w+\b', text.lower()) if 'agent' not in word]

def remove_stop_words(words):
    """Elimina las palabras vacías (stopwords) de una lista de palabras."""
    return [word for word in words if word not in STOP_WORDS]

# ========== Análisis ==========

def analyze_discussions(df):
    """Agrega las frecuencias de palabras de todas las discusiones del DataFrame."""
    counter = Counter()
    for discussion in df['discussion']:
        for utterance in discussion:
            words = remove_stop_words(tokenize(utterance))
            counter.update(words)
    return counter

# ========== Visualización UMAP ==========

def generate_umap(word_counters):
    """Genera y guarda una visualización UMAP de los embeddings de palabras."""
    # Palabras más frecuentes (top N) por grupo
    top_words_by_group = {
        group: {word for word, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:TOP_N_WORDS]}
        for group, counter in word_counters.items()
    }

    # Determinar a qué grupo(s) pertenece cada palabra
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

    # Filtrar solo palabras que tengan vector en spaCy
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

    # Colores y marcadores por grupo
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

    # Traducción de las etiquetas para la leyenda en español
    legend_es = {
        'Democrat_political': 'Demócrata (política)',
        'Republican_political': 'Republicano (política)',
        'Democrat_non_political': 'Demócrata (no política)',
        'Republican_non_political': 'Republicano (no política)',
        'Both_political': 'Ambos (política)',
        'Both_non_political': 'Ambos (no política)'
    }

    fig = plt.figure(figsize=(12, 9))
    for group_label in sorted(set(filtered_labels)):
        if group_label.startswith('Both_'):
            continue  # Omitir palabras compartidas entre varios grupos

        indices = [i for i, l in enumerate(filtered_labels) if l == group_label]
        color = color_map.get(group_label, 'gray')
        marker = marker_map.get(group_label, 'o')

        plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            s=150,  # Tamaño de marcador aumentado
            edgecolor='k',
            alpha=0.8,
            color=color,
            marker=marker,
            label=legend_es.get(group_label, group_label.replace('_', ' '))
        )

        for j in indices:
            x, y = embedding[j]
            word_text = filtered_words[j]
            print(f"{filtered_words[j]} -> {group_label}")
            plt.text(x - 0.2, y + 0.1, word_text, fontsize=20)  # Posición desplazada

    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    with PdfPages("plots/umap_word_visualization.pdf") as pdf:
        pdf.savefig(fig)
        plt.close(fig)


# ========== Carga de datos ==========

def load_data(experiments_dir):
    """Carga las conversaciones desde los archivos JSON de cada experimento."""
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
                    print(f"No se pudo cargar {file_path}: {e}")
                    continue

                agents_config = content.get('agents_config', '')
                try:
                    number = int(agents_config.split('_')[-1].replace('.csv', ''))
                except Exception:
                    continue

                initiator = 'Democrat' if number % 2 == 0 else 'Republican'
                democrat_name = 'Agent0' if initiator == 'Democrat' else 'Agent1'
                discussion = content.get("discussion", [])[1:-1]  # Quitar instrucciones iniciales y resumen final

                for agent in ['Agent0', 'Agent1']:
                    data.append({
                        'discussion': discussion,
                        'initiator': initiator,
                        'party': 'Democrat' if agent == democrat_name else 'Republican',
                        'experiment': experiment
                    })
    return pd.DataFrame(data)

# ========== Flujo principal ==========

def main():
    df = load_data('outputs/cross_partisan_conversation')
    if df.empty:
        print("No se encontraron datos.")
        return

    word_counters = defaultdict(Counter)
    for (initiator, experiment), group in df.groupby(['initiator', 'experiment']):
        key = f"{initiator}_{experiment}"
        word_counters[key] = analyze_discussions(group)

    generate_umap(word_counters)

if __name__ == "__main__":
    main()