import os
import json
import pandas as pd

experiment_type = 'h1_non_political_reverse'

# Folder containing JSON files
folder_path = f'outputs/{experiment_type}'

democrat_name = 'Agent0'
if 'reverse' in experiment_type:
    democrat_name = 'Agent1'

# Create a list to hold all data
data = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as f:
            content = json.load(f)

        agents_config = content.get('agents_config', '')
        pre = content.get('pre_questionnaire', {})
        post = content.get('post_questionnaire', {})

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
                'agent': agent,
                'party': 'Democrat' if agent == democrat_name else 'Republican',
                'Q1_pre': pre_vals.get('Q1', None),
                'Q2_pre': pre_vals.get('Q2', None),
                'Q1_post': post_vals.get('Q1', None),
                'Q2_post': post_vals.get('Q2', None),
            })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_excel = f'{experiment_type}_questionnaire_summary.xlsx'
df.to_excel(output_excel, index=False)

print(f"Excel file saved to {output_excel}")
