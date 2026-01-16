import numpy as np

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# -------------------------
# 1. Define prompt templates
# -------------------------
physics_template = """You are a very smart physics professor.
You are great at answering questions about physics in a concise and easy-to-understand manner.
When you don't know the answer to a question, you admit that you don't know.
Here is a question: {query}"""

math_template = """You are a very good mathematician.
You are great at answering math questions.
You are so good because you are able to break down hard problems into their component parts,
answer the component parts, and then put them together to answer the broader question.
Here is a question: {query}"""

prompt_templates = [physics_template, math_template]

# -------------------------
# 2. NumPy cosine similarity
# -------------------------
def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: shape (1, d) --> query embedding
    b: shape (n, d) --> prompt embeddings
    return: shape (1, n)
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

# -------------------------
# 3. Embeddings : 把prompt templates轉成向量
# -------------------------
embeddings = OpenAIEmbeddings()

prompt_embeddings = np.array(
    embeddings.embed_documents(prompt_templates)
)
print("✓ Prompt embeddings shape:", prompt_embeddings.shape)

# -------------------------
# 4. Semantic router
# -------------------------
@chain
def prompt_router(query: str):
    query_embedding = np.array(
        embeddings.embed_query(query)
    ).reshape(1, -1)
    print("✓ Query embedding shape:", query_embedding.shape)
    similarity = cosine_similarity_np(query_embedding, prompt_embeddings)[0]
    print("✓ Similarity scores:", similarity)
    most_similar_idx = similarity.argmax()
    print("✓ Most similar index:", most_similar_idx)
    selected_template = prompt_templates[most_similar_idx]
    print("✓ Selected template:", selected_template)

    print("✓ Using MATH" if selected_template == math_template else "✓ Using PHYSICS")

    return PromptTemplate.from_template(selected_template)

# -------------------------
# 5. Full chain
# -------------------------
semantic_router = (
    prompt_router
    | ChatOpenAI()
    | StrOutputParser()
)

# -------------------------
# 6. Run
# -------------------------
result = semantic_router.invoke("What's a black hole")
print("\n✓ Semantic router result:\n", result)