from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# Initialize chat model
model = ChatOpenAI()


# Define state type
# BaseMessage https://v03.api.js.langchain.com/classes/_langchain_core.messages.BaseMessage.html
class State(TypedDict):
    # BaseMessage : 這表示 messages 是一個可以包含任何訊息類型的 list。如果寫成 list[HumanMessage] 就只能放 HumanMessage，但用 list[BaseMessage] 就可以混合放 HumanMessage、AIMessage、SystemMessage 等等
    messages: Annotated[list[BaseMessage], add_messages]


# Define prompts
# 寫文章的系統提示
generate_prompt = SystemMessage(
    "You are an essay assistant tasked with writing excellent 3-paragraph essays."
    " Generate the best essay possible for the user's request."
    " If the user provides critique, respond with a revised version of your previous attempts."
)

# 反思的系統提示
reflection_prompt = SystemMessage(
    "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
    " Provide detailed recommendations, including requests for length, depth, style, etc."
)


def generate(state: State) -> State:
    print("🤖 Generating essay with messages:", state["messages"])
    print('-------')
    # system prompt + user messages
    answer = model.invoke([generate_prompt] + state["messages"])
    return {"messages": [answer]}


def reflect(state: State) -> State:
    print("🤖 state before reflect:", state["messages"])
    # Invert the messages to get the LLM to reflect on its own output
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [reflection_prompt, state["messages"][0]] + [
        cls_map[msg.__class__](content=msg.content) for msg in state["messages"][1:]
    ]
    print("🤖 Reflecting on essay with messages:", translated)
    print('-------')
    answer = model.invoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=answer.content)]}


def should_continue(state: State):
    if len(state["messages"]) > 6:
        # End after 3 iterations, each with 2 messages
        return END
    else:
        return "reflect"


# Build the graph
builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue, ["reflect", END])
builder.add_edge("reflect", "generate")

graph = builder.compile()

# 儲存 agent 架構圖
# png_data = graph.get_graph().draw_mermaid_png()
# with open("a-reflection.png", "wb") as f:
#     f.write(png_data)
# print("架構圖已儲存至 a-reflection.png")


# Example usage
initial_state = {
    "messages": [
        HumanMessage(
            content="Write an essay about the relevance of 'The Little Prince' today."
        )
    ]
}

# Run the graph
for output in graph.stream(initial_state):
    message_type = "generate" if "generate" in output else "reflect"
    print("New message:", output[message_type]
          ["messages"][-1].content[:100], "...")
    print('-------')