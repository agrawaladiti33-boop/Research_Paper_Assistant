from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

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
# 2. Split text into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(full_text)

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# -----------------------------
# 4. Create embeddings
# -----------------------------

embeddings = model.encode(chunks)

print("Embeddings created successfully!")

print("Number of embeddings:", len(embeddings))

print("Size of each embedding:", len(embeddings[0]))