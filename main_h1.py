import numpy as np
import random

from polarmas_platform import Platform

NUM_EXPERIMENTS = 77

messages_distribution = np.random.randint(6, 16, size=NUM_EXPERIMENTS)
messages_distribution.sort()
messages_distribution[38] = 10
messages_distribution[39] = 11
messages_distribution.sort() # changes on each execution!

questionnaire = """
Q1. On a scale from 0 to 100, where 0 means you feel very cold or unfavorable, 50 that you do not feel particularly warm or cold toward the group and, 100 means you feel very warm or favorable, how do you feel about Republicans across the country?
Q2. On a scale from 0 to 100, where 0 means you feel very cold or unfavorable, 50 that you do not feel particularly warm or cold toward the group and, 100 means you feel very warm or favorable, how do you feel about Democrats across the country?
Reply to the two questions following the format <question identifier>: <value>. Just give the answer.
"""

experiment_description = """
We’ve randomly assigned you a partner that belongs to or leans toward your out party. Please have a conversation by chatting with them about the meaning of life.
Specifically, we are interested in you sharing what you think makes life meaningful and learning your conversation partner’s thoughts as someone that might hold different values and beliefs.
For example, survey research shows that many people mention family as the most important sources of meaning in their life. Survey research also shows that other people mention career, money, faith, friends, and hobbies as the most important source of meaning in their life. 
What do you think?. Limit your words to 50.
"""

# TO CHECK: median number of total words 269

for i in range(0, 1):
    platform = Platform(f'agents/h1/agents_config_{i}.csv',
                        questionnaire,
                        questionnaire,
                        experiment_description,
                        f'outputs/h1/agents_config_{i}')

    #platform.facilitate_discussion(messages_distribution[i])
    platform.facilitate_discussion(20)
    platform.save_run()
