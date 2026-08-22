import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print(" Semantic Search Engine Çalıştırılıyor\n")

# ADIM 1: Veriyi JSON Dosyasından Okuma
with open('documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

# Testi 1000+ doküman seviyesine çıkarmak için veriyi çoğaltma işlemi
scaled_documents = (documents * 250)[:1000]
print(f"Veri setinde toplam {len(scaled_documents)} adet makale mevcut.")

# ADIM 2: Embedding Modelini Yükleme
model = SentenceTransformer('all-MiniLM-L6-v2')

# ADIM 3: Dokümanları Vektörlere (Embedding) Dönüştürme
contents = [doc['content'] for doc in scaled_documents]

start_embed = time.time()
doc_embeddings = model.encode(contents, convert_to_tensor=False)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# Cosine Similarity hesaplamak için L2 Normalizasyonu 
faiss.normalize_L2(doc_embeddings)

# ADIM 4: FAISS Vektör İndeksini Kurma
dimension = doc_embeddings.shape[1]  # MiniLM için 384
index = faiss.IndexFlatIP(dimension) # Inner Product (Kosinüs Benzerliği)
index.add(doc_embeddings)
print(f"FAISS İndeksine {index.ntotal} adet vektör yüklendi.\n")

# ADIM 5: Anlamsal Arama (Semantic Search) Yapma
query = "Tümör ve kanser hastalarında uygulanan yeni tedavi yöntemleri"
print(f" Arama Sorgusu: '{query}'")

# Sorguyu vektörleştirip normalize etme işlemi
query_embedding = model.encode([query], convert_to_tensor=False)
query_embedding = np.array(query_embedding).astype('float32')
faiss.normalize_L2(query_embedding)

# En yakın 3 sonucu aratma (Top-3)
k = 3
distances, indices = index.search(query_embedding, k)

# ADIM 6: Sonuçları Ekrana Yazdırma
print("--- EN YAKIN SONUÇLAR (TOP 3) ---")
for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
    matched_doc = scaled_documents[idx]
    print(f"{rank}. [Skor: {score:.4f}] Başlık: {matched_doc['title']} ({matched_doc['category']})")
    print(f"   İçerik: {matched_doc['content']}\n")
