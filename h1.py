from datetime import datetime
from time import sleep

import pandas as pd
from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
import json

topic = "Describe your perfect day, focusing on personal experiences and moments of joy, without mentioning political topics."
context = "You are engaging in a discussion with a representative from the opposing political party."
#context = "You are engaging in a conversation."
number_messages_per_agent = 10
SCENARIO_NAME = 'h1/1MD_1MR_RR'
SCENARIO_CONFIG = pd.read_csv(f'scenarios/{SCENARIO_NAME}.csv')

output_data = {'pre_questionnaire': {}, 'post_questionnaire': {}, 'discussion': [], 'agents': {}}

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

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

memory_order = ['pre_questionnaire',
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
        while True:
            try:
                response = llm([SystemMessage(content=f"You are {self.name}, {self.role}.{self.demographics} {self.persona}"),
                            *sum(self.memory.values(), [])])
                break
            except ResourceExhausted:
                sleep(60)
        self.memory[conversation_id].append(response)
        return response.content

    def observe(self, message, conversation_id):
        self.memory[conversation_id] = [HumanMessage(content=message)]

all_agents = []
discussing_agents = []

for agent_config in SCENARIO_CONFIG.itertuples():
    agent_name = f'Agent{len(all_agents)}'
    agent = Agent(agent_name, agent_config.role, agent_config.persona, agent_config.demographics)
    all_agents.append(agent)
    output_data['agents'][agent_name] = {'role': agent_config.role,
                                         'persona': agent_config.persona,
                                         'demographics': agent_config.demographics}
    if agent_config.is_observer == False:
        discussing_agents.append(agent)

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
    f'{context} {topic} Limit your speak up to 100 words. Everytime you respond, respond with <name>:.')

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f'outputs/{SCENARIO_NAME}_#messages={number_messages_per_agent}_{timestamp}.json', 'w') as fp:
    json.dump(output_data, fp)
