import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

agents_roles_map = {
    'Agent0': 'Moderate Republican',
    'Agent1': 'Extreme Republican',
    'Agent2': 'Moderate Democrat',
    'Agent3': 'Non partisan'}

def love_question(agent_name):
    if 'Republican' in agent_name:
        return 'Q1'
    elif 'Democrat' in agent_name:
        return 'Q2'

def hate_question(agent_name):
    if 'Republican' in agent_name:
        return 'Q4'
    elif 'Democrat' in agent_name:
        return 'Q3'

input_file = "outputs/summary/h4_base_questionnaire_summary.xlsx"
df = pd.read_excel(input_file)
df = df.replace({'agent_name': agents_roles_map})
pre_polarization_level = []
post_polarization_level = []
pre_closest_group = []
post_closest_group = []

for i, row in df.iterrows():
    if 'Non partisan' == row['agent_name']:
        pre_polarization_level.append(0)
        post_polarization_level.append(0)
    else:
        love_q = love_question(row['agent_name'])
        hate_q = hate_question(row['agent_name'])
        pre_polarization_level.append(row[f'pre{love_q}'] + row[f'pre{hate_q}'])
        post_polarization_level.append(row[f'post{love_q}'] + row[f'post{hate_q}'])
    if row[f'pre{love_question("Republican")}'] > row[f'pre{love_question("Democrat")}']:
        pre_closest_group.append('Republican party')
    else:
        pre_closest_group.append('Democratic party')
    if row[f'post{love_question("Republican")}'] > row[f'post{love_question("Democrat")}']:
        post_closest_group.append('Republican party')
    else:
        post_closest_group.append('Democratic party')

df['pre_polarization_level'] = pre_polarization_level
df['post_polarization_level'] = post_polarization_level
df['pre_closest_group'] = pre_closest_group
df['post_closest_group'] = post_closest_group

df['polarization_change'] = df['post_polarization_level'] - df['pre_polarization_level']

print('Polarization change by group, mean: ')
print(df.groupby('agent_name')['polarization_change'].mean())

print('Polarization change by group, description: ')
print(df.groupby('agent_name')['polarization_change'].describe())

print('Polarization change by group, max: ')
print(df.groupby('agent_name')['polarization_change'].max())

print('Polarization change by group, min: ')
print(df.groupby('agent_name')['polarization_change'].min())

print('Closest group, PRE: ')
print(df[df['agent_name'] == 'Non partisan']['pre_closest_group'])

print('Closest group, POST: ')
print(df[df['agent_name'] == 'Non partisan']['post_closest_group'])