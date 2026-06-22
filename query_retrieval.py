import faiss
import json
import numpy as np

# Load FAISS index
index = faiss.read_index("data/faiss_index.index")

# Load metadata
with open("data/chunk_metadata.json", "r") as f:
    metadata = json.load(f)

print("Index and metadata loaded")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2" , local_files_only=True)


#take user query
query = input("Enter your question: ")

#convert query into embedding
#query_embedding = model.encode([query] ) #L2 distance comparison
query_embedding = model.encode([query] , normalize_embeddings = True) #cosine similarity


#search in FAISS
k = 3  # number of results
distances, indices = index.search(np.array(query_embedding), k)


#Retrieve top k chunks
print("\nTop relevant chunks:\n")

for i, idx in enumerate(indices[0]):
    chunk = metadata[idx]

    print(f"Result {i+1}:")
    print("Document:", chunk["document_id"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"][:300], "...")  # first 300 chars
    print("-" * 50)
