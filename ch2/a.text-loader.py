from langchain_community.document_loaders import TextLoader

# 讀取本地文字檔
loader = TextLoader('./test.txt', encoding="utf-8")
docs = loader.load()

print(docs)