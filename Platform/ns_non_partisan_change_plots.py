import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

# === CONFIGURATION ===
# FOLDER_PATH = "outputs/simulating_social_media_non_partisan/valid"
# agent_id = "Agent10"
FOLDER_PATH = "outputs/simulating_social_media_non_partisan_democrats/valid"
agent_id = "Agent13"
questions = ["Q1", "Q2", "Q3", "Q4"]

# === FUNCTION TO PARSE STRING RESPONSES ===
def parse_responses(response_str):
    return {line.split(":")[0].strip(): int(line.split(":")[1].strip()) for line in response_str.split("\n")}

# === LOOP THROUGH FILES ===
plot_data = []
filenames = os.listdir(FOLDER_PATH)
def extract_number(filename):
    # Match the pattern agent_config_<number>_...
    match = re.search(r'agent_config_(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # If no match, put it at the end

sorted_filenames = sorted(filenames, key=extract_number)

print(sorted_filenames)

for filename in sorted_filenames:
    if filename.endswith(".json"):
        filepath = os.path.join(FOLDER_PATH, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        pre = parse_responses(data["pre_questionnaire"][agent_id])
        post = parse_responses(data["post_questionnaire"][agent_id])

        print(filename, pre, post)

        pre_data = {q: [] for q in questions}
        post_data = {q: [] for q in questions}

        for q in questions:
            pre_data[q].append(pre[q])
            post_data[q].append(post[q])

        plot_data.append({'pre': pre_data, 'post': post_data})

num_participants = len(plot_data)
x = np.arange(num_participants)

# Prepare plot data
bar_width = 0.2
colors = ['skyblue', 'salmon', 'palegreen', 'plum']
offsets = [-1.5, -0.5, 0.5, 1.5]  # for grouped bars

plt.figure(figsize=(20, 6))

for i, q in enumerate(questions):
    pre_values = [item['pre'][q][0] for item in plot_data]
    post_values = [item['post'][q][0] for item in plot_data]
    diffs = [post - pre for pre, post in zip(pre_values, post_values)]

    median_diff = np.median(diffs)
    avg_diff = np.mean(diffs)

    print(f"{q}: Median difference = {median_diff:.2f}, Average difference = {avg_diff:.2f}")

    pos = x + offsets[i] * bar_width

    # Bar: difference
    plt.bar(pos, diffs, width=bar_width, bottom=pre_values, label=f'{q} Δ', color=colors[i], alpha=0.7)

    # Dot: pre value - change color to match bar color
    plt.scatter(pos, pre_values, color=colors[i], s=30, zorder=3)

# Initialize counters
count_Q1_gt_5 = 0
count_Q2_gt_5 = 0
count_Q1_and_Q4_gt_5 = 0
count_Q2_and_Q3_gt_5 = 0

for item in plot_data:
    post = {q: item['post'][q][0] for q in questions}

    if post['Q1'] > 5:
        count_Q1_gt_5 += 1
    if post['Q2'] > 5:
        count_Q2_gt_5 += 1

    if (post['Q1'] > 5 and post['Q4'] > 5) or (post['Q2'] > 5 and post['Q3'] > 5):
        count_Q1_and_Q4_gt_5 += 1

print(f"Participants with love(rep) Q1 > 5: {count_Q1_gt_5}")
print(f"Participants with love(dem) Q2 > 5: {count_Q2_gt_5}")
print(f"Participants with polarization (Q1 > 5 and Q4 > 5) OR (Q2 > 5 and Q3 > 5): {count_Q1_and_Q4_gt_5}")

# Axis formatting
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(x, [f'P{i+1}' for i in x], rotation=90)
plt.xlabel("Participants")
plt.ylabel("Score Change (Post - Pre)")
plt.title("Pre/Post Score Change with Pre Scores Marked")
plt.legend()
plt.tight_layout()
plt.grid(True, axis='y', linestyle=':', alpha=0.4)
plt.show()

