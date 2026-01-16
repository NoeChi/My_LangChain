from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
import os
from dotenv import load_dotenv  

load_dotenv(override=True)  
api_key = os.getenv("OPENAI_API_KEY")   

# the building blocks

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(model="gpt-3.5-turbo")

# combine them in a function
# @chain decorator adds the same Runnable interface for any function you write

# 命令式組合 : imperative composition
@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)


# use it

response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
# print(response.content)
print(response)