import os 
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

# 使用 override=True 來覆蓋已存在的環境變數
load_dotenv(override=True)

# 檢查環境變數
api_key = os.getenv("OPENAI_API_KEY")
print(f"✓ API Key 已載入 (長度: {len(api_key)}, ASCII: {api_key.isascii()})")

# temperature 設為 0 以獲得更確定性的回應
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 直接傳入字串作為提示詞
response = model.invoke("The sky is")

# response 涵蓋 content, additional_kwargs, response_metadata, usage_metadata
# print(f"最終回應: {response}")
print(f"✓content: {response.content}") 
# print(f"✓additional_kwargs: {response.additional_kwargs}")
# print(f"✓response_metadata: {response.response_metadata}") 
# print(f"✓usage_metadata: {response.usage_metadata}")