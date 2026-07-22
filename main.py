from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data/documents.txt", "r", encoding="utf-8") as f:
    documents = f.read().splitlines()

embeddings = model.encode(documents)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

np.save("embeddings.npy", embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

query = "Kanser tedavisi"
query_embedding = model.encode([query])
query_embedding = np.array(
    query_embedding,
    dtype=np.float32
)

distances, indices = index.search(
    query_embedding,
    k=2 #en benzer 2 dokümanı getir
)

for i in indices[0]:
    print(documents[i])