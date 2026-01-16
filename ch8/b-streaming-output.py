from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# 定義 State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 初始化 LLM
model = ChatOpenAI(model="gpt-4o")

# 定義節點函數
def chatbot(state: State) -> State:
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

# 建立圖
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

# 使用 stream_mode="updates"，只為輸出新產生的部分

# stream_mode="values" - 每次輸出完整狀態：
# {'messages': [HumanMessage("hi")]}
# {'messages': [HumanMessage("hi"), AIMessage("hello")]}
# {'messages': [HumanMessage("hi"), AIMessage("hello"), ToolMessage(...)]}

# stream_mode="updates" - 只輸出該節點新增的內容：
# {'agent': {'messages': [AIMessage("hello")]}}
# {'tools': {'messages': [ToolMessage(...)]}}

for chunk in graph.stream(input, stream_mode="updates"):
    print(chunk)