from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io

def process_pdf(file_stream, chunk_size=1000, chunk_overlap=200):
    """
    Reads a PDF file stream, extracts text page by page, and splits it into chunks.
    
    Args:
        file_stream: A file-like object containing the PDF.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The overlap between chunks.
        
    Returns:
        List of dicts containing 'text' and 'page' number.
    """
    reader = PdfReader(file_stream)
    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append({
                "text": text,
                "page": page_number + 1
            })

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for document in documents:
        page_chunks = text_splitter.split_text(document["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": document["page"]
            })

    return chunks, len(reader.pages)
