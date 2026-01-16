import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
print(f"✓ API Key 已載入 (長度: {len(api_key)}, ASCII: {api_key.isascii()})")

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 直接傳入 HumanMessage 作為提示詞
prompt = [HumanMessage(content="The sky is")]
# print(f"✓ 提示詞: {prompt}")

# response 涵蓋 content, additional_kwargs, response_metadata, usage_metadata
response = model.invoke(prompt)
# print(f"最終回應: {response}")
print(f"✓content: {response.content}") 
# print(f"✓additional_kwargs: {response.additional_kwargs}")
# print(f"✓response_metadata: {response.response_metadata}") 
# print(f"✓usage_metadata: {response.usage_metadata}")