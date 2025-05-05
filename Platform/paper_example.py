from polarmas_platform import Platform

questionnaire = """
Q1. On a scale from 0 to 100, where 0 means you feel very cold or unfavorable, 50 that you do not feel particularly warm or cold toward the group and, 100 means you feel very warm or favorable, how do you feel about Republicans across the country?
Q2. On a scale from 0 to 100, where 0 means you feel very cold or unfavorable, 50 that you do not feel particularly warm or cold toward the group and, 100 means you feel very warm or favorable, how do you feel about Democrats across the country?
Reply to the two questions following the format <question identifier>: <value>. Just give the answer.
"""

discussion_trigger = """We've randomly assigned you a partner that belongs to or leans toward your out party. Please have a conversation by chatting with them about climate change policies.
    Please share your views on climate change with each other.
    Limit your words to 50.
    Every time you respond, respond with <name>:."""


platform = Platform(f'agents/paper_example.csv',
                    questionnaire,
                    questionnaire,
                    discussion_trigger)

platform.facilitate_discussion(4)
platform.save_run('outputs', 'paper_example')

