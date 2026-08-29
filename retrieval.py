import faiss
import pickle
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Load FAISS database
# -----------------------------

index = faiss.read_index(
    "vector_db/research_paper.index"
)

# -----------------------------
# 2. Load chunks
# -----------------------------

with open("vector_db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Vector database loaded.")
print("Total chunks:", len(chunks))


# -----------------------------
# 3. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 4. Ask a question
# -----------------------------

question = "What is the Transformer architecture?"

# Convert question into embedding
question_embedding = model.encode([question])

# -----------------------------
# 5. Search FAISS
# -----------------------------

number_of_results = 3

distances, indices = index.search(
    question_embedding,
    number_of_results
)


# -----------------------------
# 6. Display results
# -----------------------------

print("\nRelevant chunks:\n")

for i, index_number in enumerate(indices[0]):

    print("=" * 60)

    print("RESULT", i + 1)

    print("=" * 60)

    print(chunks[index_number])

    print("\nDistance:", distances[0][i])