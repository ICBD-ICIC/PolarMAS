from polarmas_platform_old import Platform

experiment_type = 'h4_base'

questionnaire = """
 Q1. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Republican Party?
 Q2. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Democratic Party?
 Q3. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Republican Party?
 Q4. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Democratic Party?
 Reply to the four questions following the format <question identifier>: <value>. Just return the values.
 """

if 'h4_base' in experiment_type:
    experiment_description = """
    You are a Twitter user reading a thread that started with the following tweet:
    Tweet by @JoeBiden:
    "If we don’t take urgent action to address the climate emergency, our planet may never recover. We must get the climate change denier out of the White House and tackle this crisis head-on."
    Consider and respond to the full context of the conversation. Avoid repetition; introduce new angles or synthesize previous ideas.
    Reply to the thread tagging the user or users you are replying to.
    Stay under 280 characters per message.
    """

for i in range(0, 10):
    platform = Platform(f'agents/{experiment_type}/agents_config_{i}.csv',
                        questionnaire,
                        questionnaire,
                        experiment_description,
                        f'outputs/{experiment_type}/agents_config_{i}')

    platform.facilitate_discussion(15)
    platform.save_run()
