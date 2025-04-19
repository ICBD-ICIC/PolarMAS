import os
import json
import pandas as pd
import re

experiment_type = 'h4_base'

# Folder containing JSON files
folder_path = f'outputs/{experiment_type}'

data_rows = []

# Process each JSON file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract config number from agents_config
        config_path = data.get('agents_config', '')
        match = re.search(r'agents_config_(\d+)\.csv', config_path)
        config_number = int(match.group(1)) if match else None

        pre = data.get('pre_questionnaire', {})
        post = data.get('post_questionnaire', {})

        # For each agent
        for agent in pre:
            pre_answers = {k.strip(): int(v.strip()) for k, v in
                           [line.split(':') for line in pre[agent].strip().split('\n')]}
            post_answers = {k.strip(): int(v.strip()) for k, v in
                            [line.split(':') for line in post[agent].strip().split('\n')]}

            row = {
                'config_number': config_number,
                'agent_name': agent.strip()
            }
            # Add pre and post questionnaire responses
            for i in range(1, 5):
                row[f'preQ{i}'] = pre_answers.get(f'Q{i}', None)
                row[f'postQ{i}'] = post_answers.get(f'Q{i}', None)

            data_rows.append(row)

# Create DataFrame and save to Excel
output_excel = f'outputs/summary/{experiment_type}_questionnaire_summary.xlsx'
df = pd.DataFrame(data_rows)
df.to_excel(output_excel, index=False)
