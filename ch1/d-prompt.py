import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
print(f"✓ API Key 已載入 (長度: {len(api_key)}, ASCII: {api_key.isascii()})") 

# 使用 PromptTemplate 來建立提示詞範本
template = PromptTemplate.from_template('Answer the question based on the contrxt below. if the question cannot be answerer using the informantion provide, answer with "I don\'t know"' \
'\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:')

# PromptTemplate --> input_variables, input_types, partial_variables, template
# print(f"✓ 提示詞範本: {template}")
# print(f"✓ 提示詞範本的輸入變數: {template.input_variables}")
# print(f"✓ 提示詞範本的輸入變數類型: {template.input_types}")
# print(f"✓ 提示詞範本的部分變數: {template.partial_variables}")
# print(f"✓ 提示詞範本的內容: {template.template}")

# 使用範本來產生訊息
message = template.invoke({
    "context": "The sky is blue because of the way sunlight interacts with Earth's atmosphere.",
    "question": "Why is the sky blue?"
    })

print(f"✓ 產生的訊息: {message}")