import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')

plt.rcParams.update({'font.size': 20})

extension = 'non_partisan_democrats'
agent_id = "Agent13"
#extension = 'non_partisan'
#agent_id = "Agent10"


# === CONFIGURACIÓN ===
FOLDER_PATH = f"outputs/simulating_social_media_{extension}/valid"


questions = ["Q1", "Q2", "Q3", "Q4"]

# === FUNCIÓN PARA PARSEAR LAS RESPUESTAS (STRINGS) ===
def parse_responses(response_str):
    return {line.split(":")[0].strip(): int(line.split(":")[1].strip()) for line in response_str.split("\n")}

# === RECORRER LOS ARCHIVOS ===
pre_post_data = {q: {'pre': [], 'post': []} for q in questions}
post_values = []  # Para determinar el color
filenames = os.listdir(FOLDER_PATH)

def extract_number(filename):
    match = re.search(r'agent_config_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

sorted_filenames = sorted(filenames, key=extract_number)

for filename in sorted_filenames:
    if filename.endswith(".json"):
        filepath = os.path.join(FOLDER_PATH, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        pre = parse_responses(data["pre_questionnaire"][agent_id])
        post = parse_responses(data["post_questionnaire"][agent_id])

        for q in questions:
            pre_post_data[q]['pre'].append(pre[q])
            pre_post_data[q]['post'].append(post[q])

        post_values.append(post)

# === FUNCIÓN DE COLOR Y MARCADOR ===
# Define combinaciones de color y marcador con etiquetas
def determine_color_and_marker(post):
    if post["Q1"] > 5 and post["Q4"] > 5:
        return 'red', 'd', 'Republicano polarizado'
    elif post["Q2"] > 5 and post["Q3"] > 5:
        return 'darkblue', 's', 'Demócrata polarizado'
    elif post["Q1"] > 5:
        return 'pink', 'D', 'Republicano'
    elif post["Q2"] > 5:
        return 'blue', 'o', 'Demócrata'
    else:
        return 'black', 'P', 'No partidista'

# Graficar
legend_entries = {}  # Para guardar los handles y etiquetas de la leyenda aparte

for q in questions:
    plt.figure(figsize=(8, 6))

    df_plot = pd.DataFrame({
        'pre': pre_post_data[q]['pre'],
        'post': pre_post_data[q]['post']
    })
    df_plot['color'] = [determine_color_and_marker(pv)[0] for pv in post_values]
    df_plot['marker'] = [determine_color_and_marker(pv)[1] for pv in post_values]
    df_plot['label'] = [determine_color_and_marker(pv)[2] for pv in post_values]

    # Añadir jitter
    jitter_strength = 0.1
    df_plot['pre_jitter'] = df_plot['pre'] + np.random.normal(0, jitter_strength, len(df_plot))
    df_plot['post_jitter'] = df_plot['post'] + np.random.normal(0, jitter_strength, len(df_plot))

    used_labels = set()
    for idx, row in df_plot.iterrows():
        label = row['label']
        handle = plt.scatter(
            row['pre_jitter'],
            row['post_jitter'],
            color=row['color'],
            marker=row['marker'],
            s=200,
            alpha=0.8,
            edgecolor='white',
        )
        if label not in legend_entries:
            legend_entries[label] = (row['color'], row['marker'], handle)

    # Línea de referencia
    plt.plot([0, 10], [0, 10], '--', color='gray')

    # Etiquetas de los ejes (traducidas al español)
    labels = {
        "Q1": ("Pre Love(Republicano)", "Post Love(Republicano)"),
        "Q2": ("Pre Love(Demócrata)", "Post Love(Demócrata)"),
        "Q3": ("Pre Hate(Republicano)", "Post Hate(Republicano)"),
        "Q4": ("Pre Hate(Demócrata)", "Post Hate(Demócrata)")
    }

    plt.xlabel(labels[q][0])
    plt.ylabel(labels[q][1])
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"plots/{extension}_{q}_scatter.pdf", format="pdf", bbox_inches="tight")
    # plt.show()

from matplotlib.lines import Line2D

legend_elements = []
for label, (color, marker, _) in legend_entries.items():
    if marker == 'x':
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='white', label=label,
                   markerfacecolor='none', markeredgecolor=color, markersize=15, linestyle='None')
        )
    else:
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='white', label=label,
                   markerfacecolor=color, markeredgecolor='white', markersize=15, linestyle='None')
        )

fig, ax = plt.subplots(figsize=(12, 2))
ax.legend(
    handles=legend_elements,
    loc='center',
    frameon=False,
    ncol=len(legend_elements)
)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"plots/{extension}_legend.pdf", format="pdf", bbox_inches="tight")
