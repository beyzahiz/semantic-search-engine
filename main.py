from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
# SentenceTransformer bir class
# model bu classtan üretilmiş bir nesne

sentence = "Kediler süt içer."
embedding = model.encode(sentence)
print(embedding)