
import os
if os.path.exists("data/faiss_index.index"):
    print("Index already exists")
else:
    faiss.write_index(index, "data/faiss_index.index")


#Load your chunk data
import json
with open("data/processed_chunks.json", "r") as f:
    data = json.load(f)
print("Total chunks:", len(data))


#check mps gpu is available or not
import torch
print("MPS Available:", torch.backends.mps.is_available())
print(torch.backends.mps.is_built()) #checks if PyTorch was compiled with MPS enabled

#Load embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2",device="mps",local_files_only=True)

# Generate embeddings
texts = [item["text"] for item in data]
print("Generating embeddings...")
#embeddings = model.encode(texts, show_progress_bar=True ) #for L2 distance comparison
embeddings = model.encode(texts, show_progress_bar=True , normalize_embeddings=True) #for cosine similarity
print("Embeddings shape:", embeddings.shape)


#faiss is a database to store embeddings which is searchable.
import faiss
import numpy as np

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension) #for cosine similarity
#index = faiss.IndexFlatL2(dimension) #for L2 distance comparison
index.add(np.array(embeddings))

print("FAISS index created and embeddings added")

# Save FAISS index
faiss.write_index(index, "data/faiss_index.index")

# Save metadata
with open("data/chunk_metadata.json", "w") as f:
    json.dump(data, f)

print("Saved FAISS index and metadata")


