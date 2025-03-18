import pprint
from datetime import datetime

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
import json

#PARAMETERS
topic = "climate change"
context = "Environmental Protection Agency (EPA) issued an extraordinary cavalcade of pollution rule rollback"
number_messages_per_agent = 1
#define range of adjectives.
SCENARIO_NAME = '1D_1R_RR1'
SCENARIO_CONFIG = pd.read_csv(f'scenarios/{SCENARIO_NAME}.csv')

output_data = {'pre_questionare': {}, 'post_questionare': {}, 'discussion': []}

# Set up your Google API key
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Define LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Define the questionnaire
PRE_QUESTIONNAIRE = """
Q1. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Republican Party?
Q2. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Democratic Party?
Q3. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Republican Party?
Q4. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Democratic Party?
Reply to the four questions following the format <question identifier>: <value>.
"""

POST_QUESTIONNAIRE = """
Based on your opinion about the discussion you read/participated, reply to the four questions.
Q1. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Republican Party?
Q2. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Democratic Party?
Q3. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Republican Party?
Q4. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Democratic Party?
Reply to the four questions following the format <question identifier>: <value>, based on your opinion about the discussion you read/participated.
At the end, give your reasoning.
"""

PRE_QUESTIONNAIRE_ID = 0
POST_QUESTIONNAIRE_ID = 2
CONVERSATION_ID = 1

# Create agents
class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.memory = {}

    def respond(self, message, conversation_id):
        self.memory[conversation_id] = [HumanMessage(content=message)]
        response = llm([SystemMessage(content=f"You are {self.name}, {self.role}."), *sum(self.memory.values(), [])])
        self.memory[conversation_id].append(response)
        pprint.pp(self.memory)
        return response.content

    def observe(self, message, conversation_id):
        self.memory[conversation_id] = [HumanMessage(content=message)]

all_agents = []
discussing_agents = []

for agent_config in SCENARIO_CONFIG.itertuples():
    for amount in range(agent_config.amount):
        agent = Agent(f'Agent{len(all_agents)}', agent_config.role)
        all_agents.append(agent)
        if agent_config.observer != 'True':
            discussing_agents.append(agent)


# Facilitator to manage discussion
def facilitate_discussion(discussion_prompt):
    print("-- Initial Questionnaire --")
    for agent in all_agents:
        response = agent.respond(PRE_QUESTIONNAIRE, PRE_QUESTIONNAIRE_ID)
        output_data['pre_questionare'][agent.name] = response
        print(f"{agent.name}'s response: "
              f"{response}\n")

    print("-- Discussion Begins --")
    messages = [discussion_prompt]
    # for _ in range(10):
    #     agent = random.choice(discussing_agents)
    for i in range(number_messages_per_agent):
        for agent in discussing_agents:
            response = agent.respond(messages, CONVERSATION_ID)
            print(f"{agent.name}: {response}\n")
            messages.append(response)

    print("-- Post Discussion Questionnaire --")
    for agent in all_agents:
        agent.observe(messages, CONVERSATION_ID)
        output_data['discussion'] = messages
        response = agent.respond(POST_QUESTIONNAIRE, POST_QUESTIONNAIRE_ID)
        output_data['post_questionare'][agent.name] = response
        print(f"{agent.name}'s response:"
              f" {response}\n")

# Example run
facilitate_discussion(
    f'Under the context of {context}. Discuss on {topic}. Limit your speak up to 100 words. Everytime you respond, respond with <name>:.')

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f'outputs/{SCENARIO_NAME}_{timestamp}.json', 'w') as fp:
    json.dump(output_data, fp)
