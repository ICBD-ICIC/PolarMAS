import itertools
from datetime import datetime
from time import sleep

import pandas as pd
from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
import json


class Agent:
    def __init__(self, name, role, persona, demographics, is_observer):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        self.name = name
        self.role = role
        self.persona = persona
        self.demographics = demographics
        self.is_observer = is_observer
        self.memory = {}

    def respond(self, message, conversation_id):
        self.memory[conversation_id] = [HumanMessage(content=message)]
        while True:
            try:
                response = self.llm(
                    [SystemMessage(content=f"You are {self.name}, {self.role}.{self.demographics} {self.persona}"),
                     *sum(self.memory.values(), [])])
                break
            except ResourceExhausted as e:
                retry_delay = e.details[2].retry_delay.seconds
                print('Retrying in {} seconds...'.format(retry_delay))
                sleep(retry_delay)
        self.memory[conversation_id].append(response)
        return response.content

    def observe(self, message, conversation_id):
        self.memory[conversation_id] = [HumanMessage(content=message)]


class Platform:
    """
    Args:
        agents_config (dataframe): A filename to a csv with agents configuration. Has columns persona, role, demographics and is_observer.
        pre_questionnaire (string): Prompt for the pre-conversation questionnaire.
        post_questionnaire (string): Prompt for the post-conversation questionnaire.
        discussion_trigger (string): Prompt to start the discussion. May contain context and topic.
    """

    def __init__(self, agents_config, pre_questionnaire, post_questionnaire, discussion_trigger, logs_filepath):
        self.agents = self.initialize_agents(pd.read_csv(agents_config))
        self.pre_questionnaire = pre_questionnaire
        self.post_questionnaire = post_questionnaire
        self.discussion_trigger = discussion_trigger + 'Everytime you respond, respond with <name>:.'
        self.logs_filepath = logs_filepath
        self.memory_order = ['pre_questionnaire', 'discussion', 'post_questionnaire']
        self.logs = {'agents_config': agents_config, 'pre_questionnaire': {}, 'post_questionnaire': {}, 'discussion': []}

    @staticmethod
    def initialize_agents(agents_config):
        return [
            Agent(
                f'Agent{idx}',
                config.role,
                config.persona,
                config.demographics,
                config.is_observer
            )
            for idx, config in enumerate(agents_config.itertuples())
        ]


    def facilitate_discussion(self, total_number_messages):
        print("-- Initial Questionnaire --")
        for agent in self.agents:
            response = agent.respond(self.pre_questionnaire, self.memory_order.index('pre_questionnaire'))
            self.logs['pre_questionnaire'][agent.name] = response
            print(f'- {agent.name} - {agent.role} -')
            print(response)

        print("-- Discussion Begins --")
        messages = [self.discussion_trigger]
        discussing_agents = itertools.cycle(agent for agent in self.agents if not agent.is_observer)
        while len(messages) <= total_number_messages:
            agent = next(discussing_agents)
            response = agent.respond(messages, self.memory_order.index('discussion'))
            print(response)
            messages.append(response)

        print("-- Post Discussion Questionnaire --")
        for agent in self.agents:
            agent.observe(messages, self.memory_order.index('discussion'))
            self.logs['discussion'] = messages
            response = agent.respond(self.post_questionnaire, self.memory_order.index('post_questionnaire'))
            self.logs['post_questionnaire'][agent.name] = response
            print(f'- {agent.name} - {agent.role} -')
            print(response)

    def save_run(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f'{self.logs_filepath}_{timestamp}.json', 'w') as output_file:
            json.dump(self.logs, output_file)
