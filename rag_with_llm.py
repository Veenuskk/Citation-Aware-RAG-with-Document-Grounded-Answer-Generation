
#LOAD EVERYTHING

import torch
import faiss
import json
import numpy as np

# Load FAISS
index = faiss.read_index("data/faiss_index.index")

# Load metadata
with open("data/chunk_metadata.json", "r") as f:
    metadata = json.load(f)

# Load embedding model
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

#LOAD LLM

from transformers import pipeline

print(f'gpu availability : {torch.backends.mps.is_available()}')

generator = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    device="mps",
    torch_dtype=torch.float16
)

#TAKE YOUR QUESTION

query = input("Enter your question: ")


#RETRIEVE CHUNKS

query_embedding = embed_model.encode([query], normalize_embeddings=True)

# k = 3 (no. of chunks retrieved)
k = 5
#k = 10
distances, indices = index.search(np.array(query_embedding), k)
# similarity threshold check
if distances[0][0] < 0.3:
    print("\nNo sufficiently relevant information found in the documents.")
    exit()

retrieved_chunks = [metadata[i] for i in indices[0]]

#PREPARE CONTEXT WITH CITATION

context = ""

for i, chunk in enumerate(retrieved_chunks):
    context += f"Source [{i+1}] (Doc: {chunk['document_id']}, Page: {chunk['page']}):\n"
 #   context += chunk["text"] + "\n\n"    (send full chunks : lag memory)
    context += chunk["text"][:500] + "\n\n" # send only first 500 words



#CREATE PROMPT

prompt = f"""
You are a question answering assistant.

Answer ONLY from the provided context.

Rules:
- Every sentence MUST end with citation.
- Use citations exactly like [1], [2], [3]
- Do not use external knowledge.
- Keep answer concise.

Context:
{context}

Question:
{query}

Answer:
"""

#GENERATE ANSWER

#response = generator(prompt, max_new_tokens=200) => (generate 200 words by llm )
response = generator(
    prompt,
    max_new_tokens=200,
    do_sample=False, #Use greedy decoding (pick highest probability token each time, deterministic)
    temperature=0.1,#Controls randomness (0.1 = very low = more predictable/focused; higher = more creative)
    repetition_penalty=1.1#Discourages repeating the same words (1.1 = slight penalty)
)

generated_text = response[0]["generated_text"] #Extracts the generated text from the response
#response[0] → gets first element (it's a list of dictionaries)
#["generated_text"] → gets the "generated_text" key
#generated_text = prompt + LLM's answer

answer = generated_text[len(prompt):]  #Removes the prompt from the generated text

print("\nFinal Answer:\n")
print(answer)
print("\nSources:\n")

for i, chunk in enumerate(retrieved_chunks):
    print(f"[{i+1}] {chunk['document_id']}, Page {chunk['page']}")

