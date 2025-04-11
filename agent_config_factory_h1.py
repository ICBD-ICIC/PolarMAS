import json

import pandas as pd

OUTPUT_FOLDER = 'agents/h1/'

import random
from collections import Counter, defaultdict
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# PERSONAS = load_dataset("proj-persona/PersonaHub", "elite_persona")

personas_sample = []
with open('elite_personas.part1.jsonl', 'r', encoding='utf-8') as file:
    for i in range(0, 200):
        personas_sample.append(json.loads(file.readline()))
PERSONAS = pd.DataFrame(personas_sample)

# taken from https://www.census.gov/quickfacts/fact/table
# split race into race and ethnicity to fix "Hispanic or Latino" overlapping
# age was normalized to include people 18+
# education https://data.census.gov/profile/United_States?g=010XX00US#education
DEMOGRAPHIC_DISTRIBUTION = {
    'gender': {
        'male': 0.495,
        'female': 0.505},
    'race': {
        'white': 0.753,
        'black': 0.137,
        'asian': 0.064,
        'american_indian_alaska_native': 0.013,
        'native_hawaiian_pacific_islander': 0.003,
        'two_or_more_races': 0.031},
    'ethnicity': {
        'hispanic_or_latino': 0.195,
        'not_hispanic_or_latino': 0.805},
    'education': {
        'less_than_high_school': 0.103,
        'high_school_or_equivalent': 0.259,
        'some_college_no_degree': 0.189,
        'associates_degree': 0.088,
        'bachelors_degree': 0.218,
        'graduate_or_professional_degree': 0.143},
    'age_group': {
        'age_18_to_64': 0.774,
        'age_65_and_over': 0.226}
}


def generate_distribution_samples(demographic_distribution, n):
    """
    Generate n samples that match the given demographic distributions.

    Args:
        demographic_distribution (dict): A dictionary with demographic categories and their distributions.
        n (int): Number of total samples to generate.

    Returns:
        List[dict]: A list of dictionaries, each representing a sample with assigned demographic values.
    """
    results = []

    # Precompute the categories and their weighted value lists
    weighted_choices = {}
    for category, distribution in demographic_distribution.items():
        values, weights = zip(*distribution.items())
        weighted_choices[category] = (values, weights)

    for _ in range(n):
        sample = {}
        for category, (values, weights) in weighted_choices.items():
            sample[category] = random.choices(values, weights=weights, k=1)[0]
        results.append(sample)

    return results


def verify_distribution_samples(distribution_samples):
    actual_counts = defaultdict(Counter)
    for sample in distribution_samples:
        for category, value in sample.items():
            actual_counts[category][value] += 1

    for category, counts in actual_counts.items():
        print(f"Distribution for {category}:")
        for value, count in counts.items():
            print(f"  {value}: {count / n:.2%}")


def describe_demographics(sample):
    """
    Converts a demographic sample dictionary into a readable string description.

    Args:
        sample (dict): A dictionary with demographic keys and their values.

    Returns:
        str: A human-readable description of the demographics.
    """
    parts = []

    if 'gender' in sample:
        parts.append(sample['gender'])
    if 'race' in sample:
        parts.append(sample['race'].replace('_', ' '))
    if 'ethnicity' in sample:
        parts.append(sample['ethnicity'].replace('_', ' '))
    if 'education' in sample:
        education = sample['education'].replace('_', ' ')
        if education.startswith('bachelor'):
            education = "with a " + education
        else:
            education = "with " + education
        parts.append(education)

    return "Your demographics are " + ", ".join(parts) + ", and you live in the U.S."


n = 154
samples = generate_distribution_samples(DEMOGRAPHIC_DISTRIBUTION, n)
random.shuffle(samples)
verify_distribution_samples(samples)

persona_descriptions = PERSONAS['persona'].to_list()
random.shuffle(persona_descriptions)

output_index = 1

for i in range(0, n, 2):
    agents_config = [
        {
            'role': 'Democrat',
            'is_observer': False,
            'demographics': describe_demographics(samples[i]),
            'persona': persona_descriptions[i]
        },
        {
            'role': 'Republican',
            'is_observer': False,
            'demographics': describe_demographics(samples[i + 1]),
            'persona': persona_descriptions[i + 1]
        }]
    df = pd.DataFrame(agents_config)
    df.to_csv(f'{OUTPUT_FOLDER}agents_config_{output_index}.csv', index=False)
    output_index += 1
