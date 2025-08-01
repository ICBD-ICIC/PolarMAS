import os
import re

from polarmas_platform import Platform

questionnaire = """
 Q1. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Republican Party?
 Q2. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Democratic Party?
 Q3. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Republican Party?
 Q4. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Democratic Party?
 """

pre_questionnaire = questionnaire + 'Reply to the four questions following the format <question identifier>: <value>. Just return the values.'
post_questionnaire = questionnaire + 'Based on the conversation, reply to the four questions following the format <question identifier>: <value>. Just return the values.'

experiment_description = """You are a Twitter user reading a thread that started with the following tweet:
   Tweet by @JoeBiden:
   "If we don’t take urgent action to address the climate emergency, our planet may never recover. We must get the climate change denier out of the White House and tackle this crisis head-on."
   Consider and respond to the full context of the conversation. Avoid repetition; introduce new angles or synthesize previous ideas.
   You may tag @JoeBiden and/or the users <name>: you are replying to.
   Stay under 280 characters per message. Every time you respond, start with <name>:."""

total_messages = 13
configs = 50

already_valid = set()

for filename in os.listdir('outputs/simulating_social_media_non_partisan_democrats/valid'):
    match = re.search(r'agent_config_(\d+)', filename)
    if match:
        already_valid.add(int(match.group(1)))

for config_number in range(configs):
    if config_number in already_valid:
        continue

    platform = Platform(f'agents/simulating_social_media_non_partisan_democrats/agent_config_{config_number}.csv',
                        pre_questionnaire,
                        post_questionnaire,
                        experiment_description)

    platform.facilitate_discussion(total_messages)
    platform.save_run('outputs/simulating_social_media_non_partisan_democrats', f'agent_config_{config_number}')
