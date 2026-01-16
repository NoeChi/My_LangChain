from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
print(f"✓ API Key 已載入 (長度: {len(api_key)}, ASCII: {api_key.isascii()})")


# 使用 ChatPromptTemplate 來建立提示詞範本 --> 適合用在對話任務，可以指定角色 System, Human, AI
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
        ),
        ("human", "Context: {context}"),
        ("human", "Question: {question}"),
    ]
)

# ChatPromptTemplate --> input_variables, input_types, partial_variables, messages, 
# print(f"✓ 提示詞範本: {template}")
# print(f"✓ 提示詞範本的輸入變數: {template.input_variables}")
# print(f"✓ 提示詞範本的輸入變數類型: {template.input_types}")
# print(f"✓ 提示詞範本的部分變數: {template.partial_variables}")
# print(f"✓ 提示詞範本的訊息: {template.messages}")

response = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

print(f"message: {response}")