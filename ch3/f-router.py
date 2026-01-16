from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# Data model class
# BaseModel : 型別檢查（type checking）、自動資料驗證
# datasoource 欄位只能是 "python_docs" 或 "js_docs"
# Field(...) 代表必填欄位
# description: 欄位描述
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="Given a user question, choose which datasource would be most relevant for answering their question",
    )


# Prompt template
# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)

"""
with_structured_output: Model wrapper that returns outputs formatted to match the given schema.

"""
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to the appropriate data source. Based on the programming language the question is referring to, route it to the relevant data source."""
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

# Define router
router = prompt | structured_llm

# Run
question = """Why doesn't the following code work: 
from langchain_core.prompts 
import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"]) 
prompt.invoke("french") """

result = router.invoke({"question": question})
print("\nresult: ", result)

"""
Once we extracted the relevant data source, we can pass the value into another function to execute additional logic as required:
"""


def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    else:
        return "chain for js_docs"

# 根據問題輸出datasource是python_docs還是js_docs --> 再根據結果選擇不同的chain
full_chain = router | RunnableLambda(choose_route)

result = full_chain.invoke({"question": question})
print("\nChoose route: ", result)