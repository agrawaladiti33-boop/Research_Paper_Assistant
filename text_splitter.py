from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read PDF
pdf_path = "papers/research_paper.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

print("Total characters:", len(full_text))

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split text into chunks
chunks = text_splitter.split_text(full_text)

print("Number of chunks:", len(chunks))

# Show first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print("\n" + "=" * 50)
    print("CHUNK", i + 1)
    print("=" * 50)
    print(chunk)