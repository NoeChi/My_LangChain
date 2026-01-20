from typing import Literal
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# =========================
# 1. Decision schema
# =========================
class SupervisorDecision(BaseModel):
    # basemodel 用於定義結構化輸出
    next: Literal["researcher", "coder", "FINISH"]


# =========================
# 2. Models
# =========================
# 一般對話模型（給 researcher / coder 用）
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Supervisor 專用（structured output）
supervisor_llm = llm.with_structured_output(SupervisorDecision)


# =========================
# 3. State
# =========================
class AgentState(MessagesState):
    # MessagesState 已經有繼承 messages 欄位了
    # MessagesState = messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["researcher", "coder", "FINISH"]


# =========================
# 4. Supervisor node
# =========================
def supervisor(state: AgentState):
    print("\n========================")
    # 使用 supervisor_llm 來決定下一步行動，結構化輸出
    print("🧑‍💼 Supervisor evaluating state :", state)
    decision = supervisor_llm.invoke(
        [
            (
                "system",
                "You are a supervisor managing two workers: researcher and coder. "
                "Decide who should act next, or FINISH if the task is complete.",
            ),
            *state["messages"],
            (
                "system",
                "If the user request has been sufficiently addressed, choose FINISH. Otherwise choose exactly one of: researcher or coder.",
            ),
        ]
    )
    # 會輸出next欄位，決定下一步要去哪個節點
    # 會輸出 messages 欄位，傳遞給下一個節點使用

    return {
        "next": decision.next,
    }


# =========================
# 5. Researcher node
# =========================
def researcher(state: AgentState):
    response = llm.invoke(
        [
            ("system", "You are a research assistant. Provide analysis and insights."),
            state["messages"][-1],
        ]
    )
    # 只會輸出 messages 欄位，add_messages reducer 會自動合併
    return {
        "messages": [response],
    }


# =========================
# 6. Coder node
# =========================
def coder(state: AgentState):
    response = llm.invoke(
        [
            ("system", "You are a coding assistant. Write code or technical solutions."),
            state["messages"][-1],
        ]
    )
    # 只會輸出 messages 欄位，add_messages reducer 會自動合併
    return {
        "messages": [response],
    }


# =========================
# 7. Build graph
# =========================
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"researcher": "researcher", "coder": "coder", "FINISH": "__end__"},
)

builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()

# 儲存 agent 架構圖
# png_data = graph.get_graph().draw_mermaid_png()
# with open("d-supervisor.png", "wb") as f:
#     f.write(png_data)
# print("架構圖已儲存至 d-supervisor.png")

# =========================
# 8. Example run
# =========================
initial_state = {
    "messages": [
        HumanMessage(
            content="What is the history of AI technique."
            # content="what is 4!"
        )
    ],
    "next": "supervisor",
}

for step in graph.stream(initial_state):
    print("\n========================")
    print("Step output:")
    print(step)

    if "messages" in step:
        last = step["messages"][-1]
        if isinstance(last, AIMessage):
            print("\nAgent response:")
            print(last.content)
