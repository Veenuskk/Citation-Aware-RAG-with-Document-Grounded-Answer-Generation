import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        pages.append({
            "page_number": page_num + 1,
            "text": text
        })

    return pages

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())  # remove extra spaces
    return text

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(chunk_words)
        })

        start += (chunk_size - overlap)
        chunk_id += 1

    return chunks

import json
from tqdm import tqdm

DATA_PATH = "data/raw_pdfs"
OUTPUT_FILE = "data/processed_chunks.json"

all_chunks = []

for file in tqdm(os.listdir(DATA_PATH)):
    if file.endswith(".pdf"):
        doc_id = file.replace(".pdf", "")
        pdf_path = os.path.join(DATA_PATH, file)

        pages = extract_text_from_pdf(pdf_path)

        for page in pages:
            page_num = page["page_number"]
            text = clean_text(page["text"])

            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk["text"],
                    "document_id": doc_id,
                    "page": page_num,
                    "chunk_id": chunk["chunk_id"]
                })

# Save to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_chunks, f, indent=2

print("Processing complete. Saved to:", OUTPUT_FILE)