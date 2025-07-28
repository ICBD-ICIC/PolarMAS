import os
import json

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')

plt.rcParams.update({'font.size': 20})  # Applies to all elements unless overridden

# Data preparation function
def polarization_metrics(df):
    print('Out group warmth change mean: ')
    print(df['out_group_warmth_change'].mean())

    print('Out group warmth change median: ')
    print(df['out_group_warmth_change'].median())


# Experiment configurations
experiments = ['non_political', 'political']
data = []

# Data loading and processing
for experiment in experiments:
    folder_path = f'outputs/cross_partisan_conversation/{experiment}'  # Adjust path as needed

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
            identifier = os.path.splitext(filename)[0]

            for agent in ['Agent0', 'Agent1']:
                def parse_qs(text):
                    lines = text.strip().split('\n')
                    return {line.split(':')[0].strip(): int(line.split(':')[1].strip()) for line in lines}


                pre_vals = parse_qs(pre.get(agent, 'Q1: 0\nQ2: 0'))
                post_vals = parse_qs(post.get(agent, 'Q1: 0\nQ2: 0'))

                data.append({
                    'filename': filename,
                    'initiator': initiator,
                    'agent': f'{agent}_{number}',
                    'party': 'Democrat' if agent == democrat_name else 'Republican',
                    'Q1_pre': pre_vals.get('Q1', None),
                    'Q2_pre': pre_vals.get('Q2', None),
                    'Q1_post': post_vals.get('Q1', None),
                    'Q2_post': post_vals.get('Q2', None),
                    'experiment': experiment
                })

df = pd.DataFrame(data)
df['Q1_diff'] = df['Q1_post'] - df['Q1_pre']
df['Q2_diff'] = df['Q2_post'] - df['Q2_pre']
df['out_group_warmth_change'] = np.where(
    df['party'] == 'Republican',
    df['Q2_diff'],
    df['Q1_diff']
)


# Generate the plots for both political and non_political experiments
def plot_out_group_warmth_change_barplot(df, experiment_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 6))

    # Copy and prepare the DataFrame
    df_copy = df.copy()
    df_copy['party_order'] = df_copy['party'].apply(lambda x: 0 if x == 'Republican' else 1)
    df_copy['agent_num'] = df_copy['agent'].str.extract(r'_(\d+)$').astype(int)
    df_copy = df_copy.sort_values(by=['party_order', 'agent_num'])

    # Custom order and color palette
    agent_order = df_copy['agent'].tolist()
    party_colors = {'Republican': 'red', 'Democrat': 'blue'}

    # Plot
    sns.barplot(
        data=df_copy,
        x='agent',
        y='out_group_warmth_change',
        hue='party',
        dodge=False,
        order=agent_order,
        palette=party_colors
    )

    # Add zero line
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Add median line
    median_val = df_copy['out_group_warmth_change'].median()
    plt.axhline(median_val, color='green', linewidth=1.2, linestyle='-.', label=f'Median: {median_val:.2f}')

    # Aesthetics
    plt.xlabel('Agent')
    plt.ylabel('Out Group Warmth Change')
    plt.title(f'Out Group Warmth Change - {experiment_name} Experiment')
    plt.xticks(rotation=90, fontsize=6)
    plt.legend(title='Party', loc='upper right')
    #plt.tight_layout()
    plt.xticks([], [])
    plt.show()

def plot_out_group_warmth_boxplot(df, experiment_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))

    # Boxplot with Seaborn
    sns.boxplot(
        data=df,
        x='party',
        y='out_group_warmth_change',
        palette={'Republican': 'red', 'Democrat': 'blue'}
    )

    # Aesthetics
    plt.title(f'Out Group Warmth Change by Party - {experiment_name} Experiment')
    plt.xlabel('Party')
    plt.ylabel('Out Group Warmth Change')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # zero line
    plt.tight_layout()
    plt.show()

def plot_out_group_warmth_boxplot_by_agent(df, experiment_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 6))

    # Sort agents for visual clarity
    df_copy = df.copy()
    df_copy['party_order'] = df_copy['party'].apply(lambda x: 0 if x == 'Republican' else 1)
    df_copy['agent_num'] = df_copy['agent'].str.extract(r'_(\d+)$').astype(int)
    df_copy = df_copy.sort_values(by=['party_order', 'agent_num'])

    agent_order = df_copy['agent'].tolist()
    palette = {'Republican': 'red', 'Democrat': 'blue'}

    # Boxplot
    sns.boxplot(
        data=df_copy,
        x='agent',
        y='out_group_warmth_change',
        hue='party',
        dodge=False,
        order=agent_order,
        palette=palette
    )

    # Aesthetics
    plt.title(f'Out Group Warmth Change by Agent - {experiment_name} Experiment')
    plt.xlabel('Agent')
    plt.ylabel('Out Group Warmth Change')
    plt.xticks(rotation=90, fontsize=6)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend(title='Party', loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_pre_post_scatter(df, experiment_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare data
    df_copy = df.copy()
    df_copy['x_val'] = df_copy.apply(
        lambda row: row['Q1_pre'] if row['party'] == 'Democrat' else row['Q2_pre'],
        axis=1
    )
    df_copy['y_val'] = df_copy.apply(
        lambda row: row['Q1_post'] if row['party'] == 'Democrat' else row['Q2_post'],
        axis=1
    )

    # Color palette
    party_colors = {'Republican': 'red', 'Democrat': 'blue'}

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_copy,
        x='x_val',
        y='y_val',
        hue='party',
        palette=party_colors,
        s=100,
        alpha=0.8
    )

    # Add reference line y = x
    plt.plot([df_copy['x_val'].min(), df_copy['x_val'].max()],
             [df_copy['x_val'].min(), df_copy['x_val'].max()],
             linestyle='--', color='gray', label='No Change')

    # Aesthetics
    plt.xlabel('Pre-Conversation Outgroup Warmth')
    plt.ylabel('Post-Conversation Outgroup Warmth')
    plt.title(f'Pre vs Post Outgroup Warmth - {experiment_name} Experiment')
    plt.legend(title='Party')
    plt.tight_layout()
    plt.show()

def plot_pre_post_scatter_noisy(df, experiment_name):
    # Prepare data
    df_copy = df.copy()
    df_copy['x_val'] = df_copy.apply(
        lambda row: row['Q1_pre'] if row['party'] == 'Democrat' else row['Q2_pre'],
        axis=1
    )
    df_copy['y_val'] = df_copy.apply(
        lambda row: row['Q1_post'] if row['party'] == 'Democrat' else row['Q2_post'],
        axis=1
    )

    # Apply jitter to avoid overlapping points
    jitter_strength = 0.3
    df_copy['x_val_jitter'] = df_copy['x_val'] + np.random.normal(0, jitter_strength, size=len(df_copy))
    df_copy['y_val_jitter'] = df_copy['y_val'] + np.random.normal(0, jitter_strength, size=len(df_copy))

    # Color and marker palettes
    party_colors = {'Republican': 'red', 'Democrat': 'blue'}
    party_markers = {'Republican': "D", 'Democrat': "o"}  # square for Republican, circle for Democrat

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_copy,
        x='x_val_jitter',
        y='y_val_jitter',
        hue='party',
        style='party',  # Different markers per party
        palette=party_colors,
        markers=party_markers,
        s=200,
        alpha=0.8
    )

    # Add reference line y = x
    min_val = min(df_copy['x_val'].min(), df_copy['y_val'].min())
    max_val = max(df_copy['x_val'].max(), df_copy['y_val'].max())
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle='--', color='gray', label='No Change')

    # Aesthetics
    plt.xlabel('Pre-conversation out-group warmth')
    plt.ylabel('Post-conversation out-group warmth')
    plt.legend(title='Party')
    plt.tight_layout()
    plt.savefig(f"{experiment_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# Filter data for political and non_political experiments and explicitly copy the DataFrames
df_political = df[df['experiment'] == 'political'].copy()
df_non_political = df[df['experiment'] == 'non_political'].copy()

# plot_out_group_warmth_change_barplot(df_political, 'Political')
# plot_out_group_warmth_change_barplot(df_non_political, 'Non-Political')
#
# plot_out_group_warmth_boxplot(df_political, 'Political')
# plot_out_group_warmth_boxplot(df_non_political, 'Non-Political')

# plot_out_group_warmth_boxplot_by_agent(df_political, 'Political')
# plot_out_group_warmth_boxplot_by_agent(df_non_political, 'Non-Political')

# plot_pre_post_scatter(df_political, 'Political')
# plot_pre_post_scatter(df_non_political, 'Non-Political')

plot_pre_post_scatter_noisy(df_political, 'Political')
plot_pre_post_scatter_noisy(df_non_political, 'Non-Political')