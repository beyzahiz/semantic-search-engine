from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
# SentenceTransformer bir class
# model bu classtan üretilmiş bir nesne

sentences = [
    "Kediler süt içer.",
    "Yavru kediler süt içmeyi sever.",
    "Python programlama dilidir.",
    "Araba tamircisi lastiği değiştirdi."
]

embeddings = model.encode(sentences)

similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

print(similarity)

