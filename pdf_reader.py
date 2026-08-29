from pypdf import PdfReader

pdf_path = "papers/research_paper.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages):
    text = page.extract_text()

    print("\n" + "=" * 50)
    print("PAGE", page_number + 1)
    print("=" * 50)

    print(text[:1000])