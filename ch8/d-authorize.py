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

async def main():
    # 建立圖
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")
    # 使用 tools_condition 判斷是否需要呼叫工具
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")

    graph = builder.compile(checkpointer=MemorySaver())

    # 儲存 agent 架構圖
    # png_data = graph.get_graph().draw_mermaid_png()
    # with open("d-authorize.png", "wb") as f:
    #     f.write(png_data)
    # print("架構圖已儲存至 d-authorize.png")


    # 這段程式碼的流程是：
    # 當使用者輸入的問題（如 "hello"）不需要呼叫工具時，model 會直接在 chatbot 節點生成回應
    # tools_condition 判斷沒有 tool calls，會直接導向 END
    # 結果就在這個 async for 迴圈中以 chunk 形式輸出
    user_message = {
        "messages": [
            HumanMessage(
                # "hello"
                "What is the latest AI techniques?"
            )
        ]
    }

    config = {"configurable": {"thread_id": "1"}}

    # 第一次執行，會在 tools 節點前暫停
    async for chunk in graph.astream(user_message, config, interrupt_before=["tools"]):
        print(chunk)

    # 檢查是否有待執行的工具呼叫
    state = graph.get_state(config)

    if state.next:  # 如果有下一個節點要執行（即 tools）
        # 取得最後一則訊息，查看 tool calls
        last_message = state.values["messages"][-1]

        # 是否有tool_calls屬性且不為空
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("\n" + "=" * 50)
            print("待執行的工具呼叫：")
            for tool_call in last_message.tool_calls:
                print(f"  - 工具: {tool_call['name']}")
                print(f"    參數: {tool_call['args']}")
            print("=" * 50)

            # 詢問使用者是否授權
            user_input = input("\n是否授權執行？(y/n): ").strip().lower()

            if user_input == "y":
                print("\n已授權，繼續執行...\n")
                # 繼續執行 graph
                # LangGraph 看到 input 是 None，會轉向 config 找 thread_id
                async for chunk in graph.astream(None, config):
                    print(chunk)
            else:
                print("\n已拒絕，取消工具執行。")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())