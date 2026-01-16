from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

PYTHON_CODE = """ def hello_world(): print("Hello, World!") # Call the function hello_world() """

#按照程式語言邏輯分割
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)

# 原本不是document --> 先轉成documents
python_docs = python_splitter.create_documents([PYTHON_CODE])

print(python_docs)