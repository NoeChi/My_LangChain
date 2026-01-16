from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")


# 每一個 graph 由 state, node, edge 組成
# state: 定義 graph 中會傳遞的資料結構
# node: 定義 graph 中的運算單元 (function)
# edge: 定義 graph 中 node 之間的連結關係 

class State(TypedDict):
    # 定義訊息的資料結構是list
    # Annotated[基本類型, 額外的元數據 (Metadata)]
    # add_messages: 把過去的對話訊息存入 state 中
    messages: Annotated[list, add_messages]

# 建立 StateGraph 的 builder, 我要建立一個工作流，而這個工作流中所有節點共享的數據結構就是我剛剛定義的 State
builder = StateGraph(State)

model = ChatOpenAI()

# 定義 graph 中的 node (function)
def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# Add the chatbot node
# builder.add_node(節點名稱, 執行函數)
builder.add_node("chatbot", chatbot)

# Add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# Run the graph
input = {"messages": [HumanMessage("hi!")]}
for chunk in graph.stream(input):
    print(chunk)
# answer = graph.invoke(input)
# print(answer)