from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# -----------------------------
# 1. Read PDF
# -----------------------------

pdf_path = "papers/research_paper.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"


# -----------------------------
# 2. Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(full_text)

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Create embeddings
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embeddings created.")


# -----------------------------
# 4. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created.")
print("Vectors stored:", index.ntotal)


# -----------------------------
# 5. Save FAISS index
# -----------------------------

os.makedirs("vector_db", exist_ok=True)

faiss.write_index(index, "vector_db/research_paper.index")


# -----------------------------
# 6. Save chunks
# -----------------------------

with open("vector_db/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Vector database saved successfully!")