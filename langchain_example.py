import random

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os

#PARAMETERS
topic = "climate change"
context = "Environmental Protection Agency (EPA) issued an extraordinary cavalcade of pollution rule rollback."
number_messages = 10
#neutral_discuss = False

# Set up your Google API key
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Define LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Define the questionnaire
QUESTIONNAIRE = """
Q1. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Republican Party?
Q2. On a scale from 0 to 10, where 0 represents no support at all and 10 signifies unwavering support, how would you rate your level of support for the Democratic Party?
Q3. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Republican Party?
Q4. On a scale from 0 to 10, where 0 means no dislike at all and 10 represents extreme hatred, how would you rate your level of dislike for the Democratic Party?
Reply to the four questions following the format <question identifier>: <value>.
"""

#TBD: get 3 different classes for each type, and parametrize the moderate/polarized/extreme etc
# Create agents
class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.memory = []

    def respond(self, message):
        self.memory.append(HumanMessage(content=message))
        response = llm([SystemMessage(content=f"You are {self.name}, {self.role}."), *self.memory])
        self.memory.append(response)
        return response.content

# Instantiate agents
agent1 = Agent("Agent1", "Moderate democrat")
agent2 = Agent("Agent2", "Extreme republican")
agent3 = Agent("Agent3", "Neutral about democrats and republicans")

all_agents = [agent1, agent2, agent3]
discussing_agents = [agent1, agent2, agent3] #TBD remove neutrals if parameter set to tre

# Facilitator to manage discussion
def facilitate_discussion(discussion_prompt):
    print("-- Initial Questionnaire --")
    for agent in all_agents:
        print(f"{agent.name}'s response: {agent.respond(QUESTIONNAIRE)}\n")

    print("-- Discussion Begins --")
    messages = [discussion_prompt]
    for _ in range(10):
        agent = random.choice(discussing_agents)
        response = agent.respond(messages)
        print(f"{agent.name}: {response}\n")
        messages.append(response)

    print("-- Post Discussion Questionnaire --")
    for agent in all_agents:
        print(f"{agent.name}'s response: {agent.respond(QUESTIONNAIRE)}\n")

# Example run
facilitate_discussion(f'Under the context of {context}. Discuss on {topic}. Limit your speak up to 100 words.')
