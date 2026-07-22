from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Kanser tedavisinde immünoterapi umut veriyor.",
    "Python veri bilimi için çok kullanılır.",
    "Kediler süt içmeyi sever.",
    "Kalp hastalıkları erken teşhis ile önlenebilir.",
    "Makine öğrenmesi sağlık alanında kullanılmaktadır."
]

embeddings = model.encode(documents)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

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
    k=2
)