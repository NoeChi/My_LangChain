from langchain_classic.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document  # 新版用這個
# from langchain.docstore.document import Document  # 舊版寫法
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# 你的 Postgres + pgvector
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

collection_name = "my_docs"
namespace = "my_docs_namespace"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 1) 建立 PGVector vectorstore
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# 2) 建立 RecordManager（可以用同一個 Postgres 當 cache DB）
# 追蹤資料存取的紀錄
record_manager = SQLRecordManager(
    namespace,
    db_url=connection,   # 或者用專門的 sqlite: "sqlite:///record_manager_cache.sql"
)

# 建立 RecordManager 用的 table（第一次一定要叫一次）
record_manager.create_schema()

# 3) 建立文件（要記得 metadata 裡有 source_id_key）
docs = [
    Document(page_content="there are cats in the pond",
             metadata={"id": 1, "source": "cats.txt"}),
    Document(page_content="ducks are also found in the pond",
             metadata={"id": 2, "source": "ducks.txt"}),
]

# 4) 第一次 index：會「加進 vectorstore + 在 record_manager 記錄 hash」
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",  # incremental / full / None
    source_id_key="source",  # 用 metadata["source"] 當這一組文件的來源 id
)
print("Index attempt 1:", index_1)

# 5) 第二次 index 同一批 → 不會重算、不會重寫（只會 num_skipped=2）
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("Index attempt 2:", index_2)

# 6) 修改某一份內容 → 同一個 source 會被「舊的刪掉、寫入新的」
docs[0].page_content = "I just modified this document!"

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("Index attempt 3:", index_3)

# 1. 先清空dataset
# SELECT * FROM upsertion_record WHERE namespace = 'my_docs_namespace';

# DELETE FROM upsertion_record WHERE namespace = 'my_docs_namespace';

# SELECT
#     id,
#     document,
#     cmetadata,
#     collection_id
# FROM langchain_pg_embedding;


# DELETE FROM langchain_pg_embedding
# WHERE id NOT IN (
#     SELECT id
#     FROM langchain_pg_embedding
#     ORDER BY id
#     LIMIT 0
# );

# SELECT COUNT(*) FROM langchain_pg_embedding;


# 2.執行到index 1 --> 看一下langchain_pg_embedding
# SELECT
#     id,
#     document,
#     cmetadata,
#     collection_id
# FROM langchain_pg_embedding;

# 3.再看一下my_docs_namespace
# SELECT * FROM upsertion_record WHERE namespace = 'my_docs_namespace';

# 4.執行到index 3 --> 看一下langchain_pg_embedding
# SELECT
#     id,
#     document,
#     cmetadata,
#     collection_id
# FROM langchain_pg_embedding;

# 5.再看一下my_docs_namespace
# SELECT * FROM upsertion_record WHERE namespace = 'my_docs_namespace';


