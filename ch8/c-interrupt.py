from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# 定義 State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 定義工具
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# 初始化 LLM 並綁定工具
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# 定義節點函數
def chatbot(state: State) -> State:
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

def main():
    # 建立圖
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")
    # 使用 tools_condition 判斷是否需要呼叫工具
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")

    # 使用 interrupt_before 在每個節點執行前中斷，讓用戶決定是否繼續
    graph = builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["chatbot", "tools"]  # 在這些節點執行前中斷
    )

    input_data = {
        "messages": [
            HumanMessage(
                # "How old was the 30th president of the United States when he died?"
                "Search for the latest AI techniques?"
            )
        ]
    }

    config = {"configurable": {"thread_id": "1"}}

    print("=== 開始執行 graph ===\n")

    # 初始執行
    current_input = input_data

    while True:
        # 執行到下一個中斷點
        # LangGraph 看到 input 是 None，會轉向 config 找 thread_id
        for event in graph.stream(current_input, config):
            print(f"節點輸出: {event}\n")

        # 取得當前狀態
        state = graph.get_state(config)

        # 檢查是否還有下一個節點要執行
        if not state.next:
            print("=== Graph 執行完成 ===")
            break

        # 詢問用戶是否繼續
        next_node = state.next[0]
        print(f">>> 即將執行節點: {next_node}")
        user_input = input("是否繼續執行? (y/n): ").strip().lower()

        if user_input != 'y':
            print("=== 用戶選擇中斷 ===")
            break

        print(f"繼續執行 {next_node}...\n")
        # 傳入 None 表示繼續執行（使用 checkpoint 中的狀態）
        current_input = None


if __name__ == "__main__":
    main()
