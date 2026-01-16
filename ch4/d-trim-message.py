from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI

# Define sample messages
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Create trimmer
trimmer = trim_messages(
    max_tokens=65,
    strategy="last", # Keep the last messages within the token limit
    token_counter=ChatOpenAI(model="gpt-4o"),
    include_system=True, # Keep system message
    allow_partial=False, # Do not allow partial messages
    start_on="human", # Start trimming from the last human message
)
"""
[SystemMessage(content="you're a good assistant", additional_kwargs={}, 
response_metadata={}), HumanMessage(content='whats 2 + 2', additional_kwargs={}, 
response_metadata={}), AIMessage(content='4', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='thanks', additional_kwargs={}, response_metadata={}), 
AIMessage(content='no problem!', additional_kwargs={}, response_metadata={}), 
umanMessage(content='having fun?', additional_kwargs={}, response_metadata={}), 
AIMessage(content='yes!', additional_kwargs={}, response_metadata={})]
"""


# Apply trimming
trimmed = trimmer.invoke(messages)
print(trimmed)