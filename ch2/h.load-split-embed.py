from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# Load the document
loader = TextLoader("./test.txt", encoding="utf-8")
doc = loader.load()

# Split the document
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
chunks = splitter.split_documents(doc)

# Generate embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print(embeddings)
# print("embedding type:", type(embeddings))
# print("count:", len(embeddings))
# if embeddings:
#     print("vector type:", type(embeddings[0]))
#     print("vector length:", len(embeddings[0]))