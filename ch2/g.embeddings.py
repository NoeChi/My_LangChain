from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIEmbeddings(model="text-embedding-3-small")

# model = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     dimensions=512
# )

embeddings = model.embed_documents([
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])

# print(embeddings)
# print("embedding type:", type(embeddings))
# print("count:", len(embeddings))
if embeddings:
    print("vector type:", type(embeddings[0]))
    print("vector length:", len(embeddings[0]))