from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
print(f"✓ API Key 已載入 (長度: {len(api_key)}, ASCII: {api_key.isascii()})")

# both `template` and `model` can be reused many times

template = PromptTemplate.from_template("""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

# PromptTemplate --> input_variables, input_types, partial_variables, template
# print(f"✓ 提示詞範本: {template}")
# print(f"✓ 提示詞範本的輸入變數: {template.input_variables}")
# print(f"✓ 提示詞範本的輸入變數類型: {template.input_types}")
# print(f"✓ 提示詞範本的部分變數: {template.partial_variables}")
# print(f"✓ 提示詞範本的內容: {template.template}")

model = ChatOpenAI(model="gpt-3.5-turbo")

# `prompt` and `completion` are the results of using template and model once

prompt = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

response = model.invoke(prompt)
# print(f"最終回應: {response}")
print(f"✓ 模型回應: {response.content}")