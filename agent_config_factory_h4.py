import pandas as pd
import random
from collections import Counter, defaultdict

OUTPUT_FOLDER = 'agents/h4_base/'
PERSONAS = pd.read_csv('personas/persona_descriptions.csv')

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
            print(f"  {value}: {count / len(distribution_samples):.2%}")


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


runs = 10
agents = 4
samples = generate_distribution_samples(DEMOGRAPHIC_DISTRIBUTION, runs*agents)
random.shuffle(samples)
verify_distribution_samples(samples)

shuffled_descriptions = {
    group: iter(group_df.sample(frac=1, random_state=42)["description"].tolist())
    for group, group_df in PERSONAS.groupby("group")
}

output_index = 0

for i in range(0, runs*agents, agents):
    agents_config = [
        {
            'role': 'Moderate Republican',
            'is_observer': False,
            'demographics': describe_demographics(samples[i]),
            'persona': next(shuffled_descriptions['Republican'])
        },
        {
            'role': 'Extreme Republican',
            'is_observer': False,
            'demographics': describe_demographics(samples[i + 1]),
            'persona': next(shuffled_descriptions['Republican'])
        },
        {
            'role': 'Moderate Democrat',
            'is_observer': False,
            'demographics': describe_demographics(samples[i + 2]),
            'persona': next(shuffled_descriptions['Democrat'])
        },
        {
            'role': 'Non partisan',
            'is_observer': True,
            'demographics': describe_demographics(samples[i + 3]),
            'persona': next(shuffled_descriptions['Neutral'])
        }
    ]
    df = pd.DataFrame(agents_config)
    df.to_csv(f'{OUTPUT_FOLDER}agents_config_{output_index}.csv', index=False)
    output_index += 1
