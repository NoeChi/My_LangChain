from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
print(f"✓ API Key 已載入 (長度: {len(api_key)}, ASCII: {api_key.isascii()})")      

# 定義結構化輸出模型
class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""

    answer: str
    """The answer to the user's question"""
    justification: str
    """Justification for the answer"""

llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# 建立具有結構化輸出的 LLM
structured_llm = llm.with_structured_output(AnswerWithJustification)

response = structured_llm.invoke("Is the sky blue?")

print(f"最終回應: {response}")