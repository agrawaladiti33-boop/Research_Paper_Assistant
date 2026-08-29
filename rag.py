import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

# ==========================================
# 2. Initialize Groq
# ==========================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================================
# 3. Load FAISS database
# ==========================================

index = faiss.read_index(
    "vector_db/research_paper.index"
)

# ==========================================
# 4. Load chunks + page numbers
# ==========================================

with open("vector_db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ==========================================
# 5. Load embedding model
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================
# 6. Ask user a question
# ==========================================

question = input(
    "\nAsk a question about the research paper: "
)

# ==========================================
# 7. Convert question into embedding
# ==========================================

question_embedding = embedding_model.encode(
    [question]
)

# ==========================================
# 8. Retrieve relevant chunks
# ==========================================

number_of_results = 3

distances, indices = index.search(
    question_embedding,
    number_of_results
)

retrieved_chunks = []

for index_number in indices[0]:

    retrieved_chunks.append(
        chunks[index_number]
    )

# ==========================================
# 9. Create context
# ==========================================

context = "\n\n".join(
    chunk["text"]
    for chunk in retrieved_chunks
)

# ==========================================
# 10. Create prompt
# ==========================================

prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the
research paper context provided below.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say that the information is not available
   in the provided paper context.
4. Give a clear and concise answer.

Research Paper Context:
-----------------------
{context}
-----------------------

User Question:
{question}
"""

# ==========================================
# 11. Ask Groq
# ==========================================

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response.choices[0].message.content

# ==========================================
# 12. Display answer
# ==========================================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(answer)

# ==========================================
# 13. Display sources
# ==========================================

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

source_pages = set()

for chunk in retrieved_chunks:

    source_pages.add(
        chunk["page"]
    )

for page in sorted(source_pages):

    print(f"📄 Page {page}")