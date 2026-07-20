from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
# SentenceTransformer bir class
# model bu classtan üretilmiş bir nesne

sentence = "Kediler süt içer."
embedding = model.encode(sentence)
#print(embedding)

sentences = [
    "Kediler süt içer.",
    "Köpekler kemik yemeyi sever.",
    "Python popüler bir programlama dilidir.",
    "Yapay zeka sağlık alanında kullanılmaktadır.",
    "Kanser tedavisinde immünoterapi umut veriyor."
]

embeddings = model.encode(sentences)


