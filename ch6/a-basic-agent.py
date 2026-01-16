import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# agent的特色：思維鏈、工具使用

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query) # 使用ast.literal_eval來安全地評估數學表達式，不會執行惡意的prompt code


search = DuckDuckGoSearchRun() # 使用DuckDuckGo搜尋工具，封裝好的搜尋工具
tools = [search, calculator] # 定義可用的工具列表
model = ChatOpenAI(temperature=0.1).bind_tools(tools) # 定義使用的語言模型並綁定工具


class State(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: State) -> State:
    print("🤖 Invoking model with messages:", state["messages"])
    print('-------')
    res = model.invoke(state["messages"])
    # print("🤖 Model response:", res)
    return {"messages": res}


builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
# consitional_edges + edge to create a loop
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

# 儲存 agent 架構圖
# png_data = graph.get_graph().draw_mermaid_png()
# with open("a-agent_graph.png", "wb") as f:
#     f.write(png_data)
# print("架構圖已儲存至 a-agent_graph.png")

# Example usage

input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

for c in graph.stream(input):
    print(c)
    print('-------')