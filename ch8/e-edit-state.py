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


def display_state_info(state, next_node: str):
    """顯示當前狀態和即將執行的節點資訊"""
    print("\n" + "=" * 50)
    print(f"即將執行節點: {next_node}")

    if state.values.get("messages"):
        last_message = state.values["messages"][-1]

        # 如果是要執行 tools 節點，顯示工具呼叫資訊
        if next_node == "tools" and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("\n待執行的工具呼叫：")
            for tool_call in last_message.tool_calls:
                print(f"  - 工具: {tool_call['name']}")
                print(f"    參數: {tool_call['args']}")

        # 如果是要執行 chatbot 節點，顯示當前訊息
        elif next_node == "chatbot":
            print(f"\n當前訊息內容: {last_message.content[:200]}..." if len(str(last_message.content)) > 200 else f"\n當前訊息內容: {last_message.content}")

    print("=" * 50)


async def main():
    # 建立圖
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")

    # 使用 interrupt_before 在每個節點執行前中斷
    graph = builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["chatbot", "tools"]
    )

    config = {"configurable": {"thread_id": "1"}}

    # 初始輸入
    input_data = {
        "messages": [
            HumanMessage("Search for the latest AI techniques?")
        ]
    }

    print("開始執行 Graph...")
    print(f"初始問題: {input_data['messages'][0].content}")

    # 第一次執行，啟動 graph
    # 輸出中斷點
    async for chunk in graph.astream(input_data, config):
        print(f"[輸出] {chunk}")

    # 持續檢查並處理每個中斷點
    while True:
        state = graph.get_state(config)

        # 如果沒有下一個節點，表示執行完成
        if not state.next:
            print("\n" + "=" * 50)
            print("Graph 執行完成！")
            if state.values.get("messages"):
                final_message = state.values["messages"][-1]
                print(f"\n最終回應:\n{final_message.content}")
            print("=" * 50)
            break

        next_node = state.next[0]
        display_state_info(state, next_node)

        # 詢問用戶授權
        print("\n選項:")
        print("  [y] 授權繼續執行")
        print("  [n] 拒絕並結束")
        print("  [e] 編輯/輸入新的 prompt")

        user_input = input("\n請選擇 (y/n/e): ").strip().lower()

        if user_input == "y":
            print("\n已授權，繼續執行...\n")
            # 繼續執行下一步
            async for chunk in graph.astream(None, config):
                print(f"[輸出] {chunk}") # 回到 while 迴圈開頭

        elif user_input == "e":
            # 讓用戶輸入新的 prompt
            new_prompt = input("\n請輸入新的指令: ").strip()

            if new_prompt:
                # 使用 update_state 更新狀態，加入新的 HumanMessage
                # as_node 參數指定這個更新要被視為從哪個節點發出
                #
                # 重要：使用 RemoveMessage 清除之前的訊息，避免殘留未完成的 tool_calls
                # 這樣可以避免 OpenAI API 錯誤：
                # "An assistant message with 'tool_calls' must be followed by tool messages"
                from langchain_core.messages import RemoveMessage

                # 取得當前所有訊息，並建立移除指令
                current_messages = state.values.get("messages", [])
                remove_messages = [RemoveMessage(id=msg.id) for msg in current_messages]

                # 先清除所有訊息，再加入新的 HumanMessage
                graph.update_state(
                    config,
                    {"messages": remove_messages + [HumanMessage(content=new_prompt)]},
                    as_node="__start__"  # 視為從起點發出，重新開始流程
                )
                print(f"\n已更新指令為: {new_prompt}")
                print("回到中斷點檢查...\n")
                # 不在這裡呼叫 astream，讓 while 迴圈回到開頭
                # 這樣會重新檢查 state.next，並正確顯示中斷詢問
                continue
            else:
                print("未輸入新指令，維持原狀態")

        else:  # 'n' 或其他輸入
            print("\n已拒絕，結束執行。")
            break


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())