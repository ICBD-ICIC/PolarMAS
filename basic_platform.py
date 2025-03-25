from datetime import datetime

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
import json

#PARAMETERS
topic = "climate change"
context = "Environmental Protection Agency (EPA) issued an extraordinary cavalcade of pollution rule rollback"
number_messages_per_agent = 1
#SCENARIO_NAME = '1MD_1MR_1N_RR1'
SCENARIO_NAME = '1ED_1MR_RR1'
SCENARIO_CONFIG = pd.read_csv(f'scenarios/{SCENARIO_NAME}.csv')

output_data = {'pre_questionnaire': {}, 'post_questionnaire': {}, 'discussion': []}

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# AGENT_KNOWLEDGE = """
# An Extreme Democrat has an unwavering support for the Democratic Party and extreme hatred towards the Republican Party.
# An Extreme Republican has an unwavering support for the Republican Party and extreme hatred towards the Democratic Party.
# """

AGENT_KNOWLEDGE = ""

PRE_QUESTIONNAIRE = """
Q1. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Republican Party?
Q2. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Democratic Party?
Q3. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Republican Party?
Q4. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Democratic Party?
Reply to the four questions following the format <question identifier>: <value>. Just give the answer.
"""

POST_QUESTIONNAIRE = """
Based on your opinion about the discussion you read/participated, reply to the four questions.
Q1. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Republican Party?
Q2. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Democratic Party?
Q3. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Republican Party?
Q4. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Democratic Party?
Reply to the four questions following the format <question identifier>: <value>. Just give the answer.
"""

memory_order = ['agent_knowledge',
                'pre_questionnaire',
                'discussion',
                'post_questionnaire']

class Agent:
    def __init__(self, name, role, persona, demographics):
        self.name = name
        self.role = role
        self.persona = persona
        self.demographics = demographics
        self.memory = {}

    def respond(self, message, conversation_id):
        self.memory[conversation_id] = [HumanMessage(content=message)]
        response = llm([SystemMessage(content=f"You are {self.name}, {self.role}.{self.demographics} {self.persona}"),
                        *sum(self.memory.values(), [])])
        self.memory[conversation_id].append(response)
        return response.content

    def observe(self, message, conversation_id):
        self.memory[conversation_id] = [HumanMessage(content=message)]

all_agents = []
discussing_agents = []

for agent_config in SCENARIO_CONFIG.itertuples():
    agent = Agent(f'Agent{len(all_agents)}', agent_config.role, agent_config.persona, agent_config.demographics)
    agent.observe(AGENT_KNOWLEDGE, memory_order.index('agent_knowledge'))
    all_agents.append(agent)
    if agent_config.is_observer == False:
        discussing_agents.append(agent)


# Facilitator to manage discussion
def facilitate_discussion(discussion_prompt):
    print("-- Initial Questionnaire --")
    for agent in all_agents:
        response = agent.respond(PRE_QUESTIONNAIRE, memory_order.index('pre_questionnaire'))
        output_data['pre_questionnaire'][agent.name] = response
        print(f'- {agent.name} - {agent.role} -')
        print(response)

    print("-- Discussion Begins --")
    messages = [discussion_prompt]
    for i in range(number_messages_per_agent):
        for agent in discussing_agents:
            response = agent.respond(messages, memory_order.index('discussion'))
            print(response)
            messages.append(response)

    print("-- Post Discussion Questionnaire --")
    for agent in all_agents:
        agent.observe(messages, memory_order.index('discussion'))
        output_data['discussion'] = messages
        response = agent.respond(POST_QUESTIONNAIRE, memory_order.index('post_questionnaire'))
        output_data['post_questionnaire'][agent.name] = response
        print(f'- {agent.name} - {agent.role} -')
        print(response)

# Example run
facilitate_discussion(
    f'Under the context of {context}. Discuss on {topic}. Limit your speak up to 100 words. Everytime you respond, respond with <name>:.')

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f'outputs/{SCENARIO_NAME}_{timestamp}.json', 'w') as fp:
    json.dump(output_data, fp)
