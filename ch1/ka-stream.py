from langchain_core.runnables import chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv  

load_dotenv(override=True)  
api_key = os.getenv("OPENAI_API_KEY") 

model = ChatOpenAI(model="gpt-3.5-turbo")


template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

# https://gemini.google.com/share/032c759c21ba
# 命令式組合 + 同步串流模式 --> 可邊產生邊回傳
@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token


for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
    print(part.content)