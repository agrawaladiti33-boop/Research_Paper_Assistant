import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

from prompts import (
    SUMMARY_PROMPT,
    CONTRIBUTIONS_PROMPT,
    LIMITATIONS_PROMPT,
    FUTURE_WORK_PROMPT
)

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
# 4. Load document chunks
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
# 6. Function to retrieve relevant chunks
# ==========================================

def retrieve_context(query, number_of_results=5):

    query_embedding = embedding_model.encode(
        [query]
    )

    distances, indices = index.search(
        query_embedding,
        number_of_results
    )

    retrieved_chunks = []

    for index_number in indices[0]:
        retrieved_chunks.append(
            chunks[index_number]
        )

    return "\n\n".join(retrieved_chunks)


# ==========================================
# 7. Function to ask Groq
# ==========================================

def ask_groq(prompt):

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ==========================================
# 8. Generate Summary
# ==========================================

def generate_summary():

    context = retrieve_context(
        "research paper summary methodology results conclusion"
    )

    prompt = SUMMARY_PROMPT.format(
        context=context
    )

    return ask_groq(prompt)


# ==========================================
# 9. Generate Contributions
# ==========================================

def generate_contributions():

    context = retrieve_context(
        "main contributions proposed method important innovations"
    )

    prompt = CONTRIBUTIONS_PROMPT.format(
        context=context
    )

    return ask_groq(prompt)


# ==========================================
# 10. Generate Limitations
# ==========================================

def generate_limitations():

    context = retrieve_context(
        "limitations weaknesses constraints disadvantages"
    )

    prompt = LIMITATIONS_PROMPT.format(
        context=context
    )

    return ask_groq(prompt)


# ==========================================
# 11. Generate Future Work
# ==========================================

def generate_future_work():

    context = retrieve_context(
        "future work future research directions improvements"
    )

    prompt = FUTURE_WORK_PROMPT.format(
        context=context
    )

    return ask_groq(prompt)


# ==========================================
# 12. Test the functions
# ==========================================

print("\n===== SUMMARY =====\n")
print(generate_summary())

print("\n===== KEY CONTRIBUTIONS =====\n")
print(generate_contributions())

print("\n===== LIMITATIONS =====\n")
print(generate_limitations())

print("\n===== FUTURE WORK =====\n")
print(generate_future_work())