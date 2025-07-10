import numpy as np
from polarmas_platform_mistral import Platform

#experiment = 'political'
experiment = 'non_political'

# Creates a median of 9 messages per run while allowing some variation on the amount per run
np.random.seed(1234)
lower_half = np.random.randint(6, 11, size=38).tolist()
upper_half = np.random.randint(11, 17, size=38).tolist()
messages_distribution = lower_half + [10] + upper_half
messages_distribution.sort()

# questionnaire = """
# Q1. On a scale from 0 to 100, where 0 means you feel very cold or unfavorable, 50 that you do not feel particularly warm or cold toward the group and, 100 means you feel very warm or favorable, how do you feel about Republicans across the country?
# Q2. On a scale from 0 to 100, where 0 means you feel very cold or unfavorable, 50 that you do not feel particularly warm or cold toward the group and, 100 means you feel very warm or favorable, how do you feel about Democrats across the country?
# Reply using only the following format and nothing else:
#
# Q1: [number]
# Q2: [number]
#
# Do not include any explanation or additional text. This is a strict-format survey. Any additional text will be invalid.
# """

questionnaire = """Answer the following two questions. For each, respond with a number from 0 to 100, where:
- 0 = very cold or unfavorable feelings
- 50 = neutral
- 100 = very warm or favorable feelings

Q1. How do you feel about Republicans across the country?  
Q2. How do you feel about Democrats across the country?

Reply using only this exact format (no explanations or extra text):

Q1: number  
Q2: number

Any other output will be invalid. Do not include any commentary or explanation. Your answer must contain at most 15 characters.
"""

if experiment == 'non_political':
    experiment_description = """We've randomly assigned you a partner that belongs to or leans toward your out party. Please have a conversation by chatting with them about the meaning of life.
    Specifically, we are interested in you sharing what you think makes life meaningful and learning your conversation partner's thoughts as someone that might hold different values and beliefs.
    For example, survey research shows that many people mention family as the most important sources of meaning in their life. Survey research also shows that other people mention career, money, faith, friends, and hobbies as the most important source of meaning in their life. 
    What do you think?. Limit your words to 50.
    Every time you respond, respond with <name>:."""

if experiment == 'political':
    experiment_description = """We've randomly assigned you a partner that belongs to or leans toward your out party. Please have a conversation by chatting with them about immigration policies.
    Specifically, we are interested in you sharing what you think makes life meaningful and learning your conversation partner's thoughts as someone that might hold different values and beliefs.
    For example, survey research shows that some people believe that stronger enforcement of our immigration laws is more important than creating a way for immigrants already here illegally to become citizens, while others believe the opposite. 
    Survey research also shows that some people believe that undocumented immigrants currently living in the U.S. are more likely than U.S. citizens to commit serious crimes, while others believe they are not.
    Please share your views on immigration policy with each other.
    Limit your words to 50.
    Every time you respond, respond with <name>:."""

for i in range(19, 77):
    platform = Platform(f'agents/cross_partisan_conversation/agents_config_{i}.csv',
                        questionnaire,
                        questionnaire,
                        experiment_description)

    platform.facilitate_discussion(messages_distribution[i])
    platform.save_run(f'outputs/cross_partisan_conversation_mistral/{experiment}', f'agents_config_{i}')
