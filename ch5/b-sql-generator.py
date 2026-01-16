from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# useful to generate SQL query
model_low_temp = ChatOpenAI(temperature=0.1)
# useful to generate natural language outputs
model_high_temp = ChatOpenAI(temperature=0.7)

# 定義 state --> State, Input, Output
class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    user_query: str
    # output
    sql_query: str
    sql_explanation: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    sql_query: str
    sql_explanation: str


generate_prompt = SystemMessage(
    "You are a helpful data analyst, who generates SQL queries for users based on their questions."
)

# 定義 node --> generate_sql, explain_sql
# State --> messafes, user_query, sql_query, sql_explanation
def generate_sql(state: State) -> State:
    # print("Generating SQL for user query:")
    # print(state)
    # print("-----")
    user_message = HumanMessage(state["user_query"])
    # * 會自動 unpack list

    # 合併system prompt 和過去訊息串
    messages = [generate_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)
    return {
        "sql_query": res.content,
        # update conversation history
        "messages": [user_message, res],
    }


explain_prompt = SystemMessage(
    "You are a helpful data analyst, who explains SQL queries to users."
)

# State --> messafes, user_query, sql_query, sql_explanation
def explain_sql(state: State) -> State:
    # print("Generating explanation for SQL query:")
    # print(state)
    # print("-----")

    # 合併system prompt 和過去訊息串
    messages = [
        explain_prompt,
        # contains user's query and SQL query from prev step
        *state["messages"],
    ]
    res = model_high_temp.invoke(messages)
    return {
        "sql_explanation": res.content,
        # update conversation history
        "messages": res,
    }

# 建立 StateGraph 的 builder
# input, output 是舊的參數名稱，現在改成 input_schema, output_schema
builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

graph = builder.compile()

# Example usage
result = graph.invoke({"user_query": "What is the total sales for each product?"})
# print(result)
print(result["sql_query"])
print(result["sql_explanation"])