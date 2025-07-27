import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')

plt.rcParams.update({'font.size': 16})
sns.set(style="whitegrid")


# === CONFIGURATION ===
# FOLDER_PATH = "outputs/simulating_social_media_non_partisan_democrats/valid"
# agent_id = "Agent13"
FOLDER_PATH = "outputs/simulating_social_media_non_partisan/valid"
agent_id = "Agent10"

questions = ["Q1", "Q2", "Q3", "Q4"]

# === FUNCTION TO PARSE STRING RESPONSES ===
def parse_responses(response_str):
    return {line.split(":")[0].strip(): int(line.split(":")[1].strip()) for line in response_str.split("\n")}

# === LOOP THROUGH FILES ===
pre_post_data = {q: {'pre': [], 'post': []} for q in questions}
post_values = []  # For color determination
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

# === COLOR AND MARKER FUNCTION ===
# Define color and marker combinations with labels
def determine_color_and_marker(post):
    if post["Q1"] > 5 and post["Q4"] > 5:
        return 'darkred', 'D', 'Polarized Republican'
    elif post["Q2"] > 5 and post["Q3"] > 5:
        return 'darkblue', '8', 'Polarized Democrat'
    elif post["Q1"] > 5:
        return 'red', 'd', 'Republican'
    elif post["Q2"] > 5:
        return 'blue', 'o', 'Democrat'
    else:
        return 'black', '.', 'Non Partisan'

# Plotting
for q in questions:
    plt.figure(figsize=(8, 6))

    df_plot = pd.DataFrame({
        'pre': pre_post_data[q]['pre'],
        'post': pre_post_data[q]['post']
    })
    df_plot['color'] = [determine_color_and_marker(pv)[0] for pv in post_values]
    df_plot['marker'] = [determine_color_and_marker(pv)[1] for pv in post_values]
    df_plot['label'] = [determine_color_and_marker(pv)[2] for pv in post_values]

    # Add jitter
    jitter_strength = 0.1
    df_plot['pre_jitter'] = df_plot['pre'] + np.random.normal(0, jitter_strength, len(df_plot))
    df_plot['post_jitter'] = df_plot['post'] + np.random.normal(0, jitter_strength, len(df_plot))

    # Unique combinations for legend
    used_labels = set()
    for idx, row in df_plot.iterrows():
        label = row['label'] if row['label'] not in used_labels else None
        plt.scatter(
            row['pre_jitter'],
            row['post_jitter'],
            color=row['color'],
            marker=row['marker'],
            s=200,
            alpha=0.8,
            label=label,
            edgecolor='white',
            linewidth=0.7
        )
        used_labels.add(row['label'])

    # Reference line
    plt.plot([0, 10], [0, 10], '--', color='gray', label='No Change')

    # Custom labels
    labels = {
        "Q1": ("Pre-conversation love(Republicans)", "Post-conversation love(Republicans)"),
        "Q2": ("Pre-conversation love(Democrats)", "Post-conversation love(Democrats)"),
        "Q3": ("Pre-conversation hate(Republicans)", "Post-conversation hate(Republicans)"),
        "Q4": ("Pre-conversation hate(Democrats)", "Post-conversation hate(Democrats)")
    }

    # Set custom axis labels
    plt.xlabel(labels[q][0])
    plt.ylabel(labels[q][1])

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(False)
    plt.legend(title='Post-conversation Party', loc='best')
    plt.tight_layout()
    # plt.savefig(f"democrats_{q}_scatter.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"non_partisan_{q}_scatter.pdf", format="pdf", bbox_inches="tight")
    plt.show()
