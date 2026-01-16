"""
1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. pip install -qU langchain_postgres
3. Run the following command to start the postgres container:
   
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
4. Use the connection string below for the postgres container

"""

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
print(f"✓ API Key 已載入 (長度: {len(api_key)}, ASCII: {api_key.isascii()})")


# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt', encoding="utf-8").load()

# 切割文字
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10, chunk_overlap=2)

# Split the documents
documents = text_splitter.split_documents(raw_documents)

# Create embeddings for the documents
embeddings_model = OpenAIEmbeddings()

db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

# Query the vector store
# db.similarity_search --> 只給搜尋結果，不好串接chain，跑完直接有result
results = db.similarity_search("query", k=4)

print('result:', results)
# print('type of result:', type(results))

print("Adding documents to the vector store")

# 自訂 IDs
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
# print('ids:', ids)
# print()

db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)

print("Documents added successfully.\n Fetched documents count:",
      len(db.get_by_ids(ids)))

print("Deleting document with id", ids[1])
# 刪除指定 ID 的文件
db.delete(ids=ids)
# 或者刪除單一 ID
# db.delete(ids=[ids[1]])

print("Document deleted successfully.\n Fetched documents count:",
      len(db.get_by_ids(ids)))

# 計算資料庫中有多少資料
# SELECT COUNT(*) FROM langchain_pg_embedding;

# 查看資料表內容    
# SELECT *
# FROM langchain_pg_embedding;

# 查看資料表內容  
# SELECT
#     id,
#     document,
#     cmetadata,
#     collection_id
# FROM langchain_pg_embedding;

# 刪除所有資料
# DELETE FROM langchain_pg_embedding
# WHERE id NOT IN (
#     SELECT id
#     FROM langchain_pg_embedding
#     ORDER BY id
#     LIMIT 0
# );

# 查看資料表內容
# SELECT COUNT(*) FROM langchain_pg_embedding;
