import os
import json

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')

plt.rcParams.update({'font.size': 20})  # Se aplica a todos los elementos salvo que se sobrescriba

# Mapeo solo para visualización (no afecta la lógica interna de 'party')
PARTY_ES = {'Republican': 'Republicano', 'Democrat': 'Demócrata'}

# Función de preparación de datos
def polarization_metrics(df):
    print('Media del cambio de calidez hacia el grupo contrario: ')
    print(df['out_group_warmth_change'].mean())

    print('Mediana del cambio de calidez hacia el grupo contrario: ')
    print(df['out_group_warmth_change'].median())


# Configuración de los experimentos
experiments = ['non_political', 'political']
data = []

# Carga y procesamiento de datos
for experiment in experiments:
    folder_path = f'outputs/cross_partisan_conversation_2/{experiment}'  # Ajustar la ruta según sea necesario

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


# Generar los gráficos para los experimentos político y no político
def plot_out_group_warmth_change_barplot(df, experiment_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 6))

    # Copiar y preparar el DataFrame
    df_copy = df.copy()
    df_copy['party_order'] = df_copy['party'].apply(lambda x: 0 if x == 'Republican' else 1)
    df_copy['agent_num'] = df_copy['agent'].str.extract(r'_(\d+)$').astype(int)
    df_copy = df_copy.sort_values(by=['party_order', 'agent_num'])

    # Orden personalizado y paleta de colores
    agent_order = df_copy['agent'].tolist()
    df_copy['party_es'] = df_copy['party'].map(PARTY_ES)
    party_colors = {'Republicano': 'red', 'Demócrata': 'blue'}

    # Gráfico
    sns.barplot(
        data=df_copy,
        x='agent',
        y='out_group_warmth_change',
        hue='party_es',
        dodge=False,
        order=agent_order,
        palette=party_colors
    )

    # Línea de cero
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Línea de la mediana
    median_val = df_copy['out_group_warmth_change'].median()
    plt.axhline(median_val, color='green', linewidth=1.2, linestyle='-.', label=f'Mediana: {median_val:.2f}')

    # Estética
    plt.xlabel('Agente')
    plt.ylabel('Cambio de calidez hacia el grupo contrario')
    plt.title(f'Cambio de calidez hacia el grupo contrario - Experimento {experiment_name}')
    plt.xticks(rotation=90, fontsize=6)
    plt.legend(title='Partido', loc='upper right')
    #plt.tight_layout()
    plt.xticks([], [])
    plt.show()

def plot_out_group_warmth_boxplot(df, experiment_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))

    # Copiar y traducir etiquetas de partido solo para visualización
    df_copy = df.copy()
    df_copy['party_es'] = df_copy['party'].map(PARTY_ES)

    # Boxplot con Seaborn
    sns.boxplot(
        data=df_copy,
        x='party_es',
        y='out_group_warmth_change',
        palette={'Republicano': 'red', 'Demócrata': 'blue'}
    )

    # Estética
    plt.title(f'Cambio de calidez hacia el grupo contrario por partido - Experimento {experiment_name}')
    plt.xlabel('Partido')
    plt.ylabel('Cambio de calidez hacia el grupo contrario')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # línea de cero
    plt.tight_layout()
    plt.show()

def plot_out_group_warmth_boxplot_by_agent(df, experiment_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 6))

    # Ordenar agentes para mayor claridad visual
    df_copy = df.copy()
    df_copy['party_order'] = df_copy['party'].apply(lambda x: 0 if x == 'Republican' else 1)
    df_copy['agent_num'] = df_copy['agent'].str.extract(r'_(\d+)$').astype(int)
    df_copy = df_copy.sort_values(by=['party_order', 'agent_num'])

    agent_order = df_copy['agent'].tolist()
    df_copy['party_es'] = df_copy['party'].map(PARTY_ES)
    palette = {'Republicano': 'red', 'Demócrata': 'blue'}

    # Boxplot
    sns.boxplot(
        data=df_copy,
        x='agent',
        y='out_group_warmth_change',
        hue='party_es',
        dodge=False,
        order=agent_order,
        palette=palette
    )

    # Estética
    plt.title(f'Cambio de calidez hacia el grupo contrario por agente - Experimento {experiment_name}')
    plt.xlabel('Agente')
    plt.ylabel('Cambio de calidez hacia el grupo contrario')
    plt.xticks(rotation=90, fontsize=6)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend(title='Partido', loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_pre_post_scatter(df, experiment_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Preparar datos
    df_copy = df.copy()
    df_copy['x_val'] = df_copy.apply(
        lambda row: row['Q1_pre'] if row['party'] == 'Democrat' else row['Q2_pre'],
        axis=1
    )
    df_copy['y_val'] = df_copy.apply(
        lambda row: row['Q1_post'] if row['party'] == 'Democrat' else row['Q2_post'],
        axis=1
    )

    # Paleta de colores
    df_copy['party_es'] = df_copy['party'].map(PARTY_ES)
    party_colors = {'Republicano': 'red', 'Demócrata': 'blue'}

    # Gráfico
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_copy,
        x='x_val',
        y='y_val',
        hue='party_es',
        palette=party_colors,
        s=100,
        alpha=0.8
    )

    # Línea de referencia y = x
    plt.plot([df_copy['x_val'].min(), df_copy['x_val'].max()],
             [df_copy['x_val'].min(), df_copy['x_val'].max()],
             linestyle='--', color='gray', label='Sin cambio')

    # Estética
    plt.xlabel('Calidez pre-conversación')
    plt.ylabel('Calidez post-conversación')
    plt.title(f'Calidez hacia el grupo contrario: antes vs. después - Experimento {experiment_name}')
    plt.legend(title='Partido')
    plt.tight_layout()
    plt.show()

def plot_pre_post_scatter_noisy(df, experiment_name):
    # Preparar datos
    df_copy = df.copy()
    df_copy['x_val'] = df_copy.apply(
        lambda row: row['Q1_pre'] if row['party'] == 'Democrat' else row['Q2_pre'],
        axis=1
    )
    df_copy['y_val'] = df_copy.apply(
        lambda row: row['Q1_post'] if row['party'] == 'Democrat' else row['Q2_post'],
        axis=1
    )

    # Aplicar jitter para evitar solapamiento de puntos
    jitter_strength = 0.3
    df_copy['x_val_jitter'] = df_copy['x_val'] + np.random.normal(0, jitter_strength, size=len(df_copy))
    df_copy['y_val_jitter'] = df_copy['y_val'] + np.random.normal(0, jitter_strength, size=len(df_copy))

    # Paletas de colores y marcadores
    df_copy['party_es'] = df_copy['party'].map(PARTY_ES)
    party_colors = {'Republicano': 'red', 'Demócrata': 'blue'}
    party_markers = {'Republicano': "D", 'Demócrata': "o"}  # rombo para republicanos, círculo para demócratas

    # Gráfico
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_copy,
        x='x_val_jitter',
        y='y_val_jitter',
        hue='party_es',
        style='party_es',  # Marcadores distintos por partido
        palette=party_colors,
        markers=party_markers,
        s=200,
        alpha=0.8
    )

    # Línea de referencia y = x
    min_val = min(df_copy['x_val'].min(), df_copy['y_val'].min())
    max_val = max(df_copy['x_val'].max(), df_copy['y_val'].max())
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle='--', color='gray', label='Sin cambio')

    # Estética
    plt.xlabel('Calidez pre-conversación')
    plt.ylabel('Calidez post-conversación')
    plt.legend(title='Partido')
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_name}_2.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# Filtrar los datos para los experimentos político y no político, copiando explícitamente los DataFrames
df_political = df[df['experiment'] == 'political'].copy()
df_non_political = df[df['experiment'] == 'non_political'].copy()

# plot_out_group_warmth_change_barplot(df_political, 'Político')
# plot_out_group_warmth_change_barplot(df_non_political, 'No político')
#
# plot_out_group_warmth_boxplot(df_political, 'Político')
# plot_out_group_warmth_boxplot(df_non_political, 'No político')

# plot_out_group_warmth_boxplot_by_agent(df_political, 'Político')
# plot_out_group_warmth_boxplot_by_agent(df_non_political, 'No político')

# plot_pre_post_scatter(df_political, 'Político')
# plot_pre_post_scatter(df_non_political, 'No político')

plot_pre_post_scatter_noisy(df_political, 'Político')
plot_pre_post_scatter_noisy(df_non_political, 'No político')