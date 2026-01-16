from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader

from langchain_community.document_loaders import WebBaseLoader

# loader = WebBaseLoader('https://www.langchain.com/')
# docs = loader.load()

loader = TextLoader('./test.txt', encoding="utf-8")
docs = loader.load()

# 文字切割

# 1000 字元為一段，重疊 200 字元
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 10 字元為一段，重疊 2 字元
splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)

# 原本就是document --> split documents
splitted_docs = splitter.split_documents(docs)

print(splitted_docs)