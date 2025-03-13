import asyncio
import random

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create an OpenAI model client.
model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    # api_key="sk-...", # Optional if you have an OPENAI_API_KEY env variable set.
)

# Create the primary agent.
republican_agent = AssistantAgent(
    name='agent1',
    model_client=model_client,
    system_message=f'You identify as a republican, speak and act as one.'
                   f'On a scale from 0 to 10, you are a 9 in support of republican party. Value can be changed and overwritten based on the conversation.'
                   f'Now you are going to discuss climate change issue with another agent that is democrat.'
                   f'At the end of your message state on a scale on 0 to 10 how much do you identify/support republicans.'
)

# Create the critic agent.
democrat_agent = AssistantAgent(
    name='agent2',
    model_client=model_client,
    system_message=f'You identify as a democrat, speak and act as one.'
                   f'On a scale from 0 to 10, you are a 6 in support of democrat party. Value can be changed and overwritten based on the conversation.'
                   f'Now you are going to discuss climate change issue with another agent that is republican.'
                   f'At the end of your message state on a scale on 0 to 10 how much do you identify/support democrats.'
)

# Define a termination condition that stops the task if the critic approves.
termination_condition = MaxMessageTermination(6)

# # Create a team with the primary and critic agents.
# team = SelectorGroupChat([democrat_agent, republican_agent],
#                          model_client=model_client,
#                          selector_prompt='Select randomly the next role from {participants} to speak.',
#                          termination_condition=termination_condition,
#                          allow_repeated_speaker=True)

team = RoundRobinGroupChat([democrat_agent, republican_agent], termination_condition=termination_condition)

# Use `asyncio.run(...)` when running in a script.
# result = asyncio.run(team.run(task="Write a short poem about the fall season."))
# print(result)

asyncio.run(Console(team.run_stream(task="Discuss.")))
