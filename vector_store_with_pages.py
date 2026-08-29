from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# ==========================================
# 1. Read PDF page by page
# ==========================================

pdf_path = "papers/research_paper.pdf"

reader = PdfReader(pdf_path)

documents = []

for page_number, page in enumerate(reader.pages):

    text = page.extract_text()

    if text:
        documents.append({
            "text": text,
            "page": page_number + 1
        })

print("Pages loaded:", len(documents))


# ==========================================
# 2. Split each page into chunks
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for document in documents:

    page_chunks = text_splitter.split_text(
        document["text"]
    )

    for chunk in page_chunks:

        chunks.append({
            "text": chunk,
            "page": document["page"]
        })

print("Number of chunks:", len(chunks))


# ==========================================
# 3. Create embeddings
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(
    texts
)

print("Embeddings created.")


# ==========================================
# 4. Create FAISS index
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored:", index.ntotal)


# ==========================================
# 5. Save database
# ==========================================

os.makedirs(
    "vector_db",
    exist_ok=True
)

faiss.write_index(
    index,
    "vector_db/research_paper.index"
)


# ==========================================
# 6. Save chunks + page numbers
# ==========================================

with open(
    "vector_db/chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )

print("Vector database with page numbers saved!")