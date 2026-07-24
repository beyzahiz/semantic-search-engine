import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Model Yükleme
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Temsili Veri Seti Oluşturma (1000 Doküman)
# Gerçek senaryoyu simüle etmek için 1000 adet rastgele cümle üretiyoruz
base_sentences = [
    "Kanser tedavisinde hedefe yönelik akıllı ilaçlar kullanılıyor.",
    "Onkoloji kliniğinde yeni immunoterapi testleri başladı.",
    "Büyük dil modelleri doğal dil işleme görevlerinde yüksek başarı gösteriyor.",
    "Yapay zeka algoritmaları görüntü tanıma süreçlerini hızlandırıyor.",
    "Makine öğrenmesi ile finansal piyasa analizleri yapılıyor.",
    "Yazılım mimarilerinde mikroservis yapısı yaygınlaşıyor."
]

# 1000 adet dokümana tamamlamak için çoğaltıyoruz
documents = (base_sentences * 170)[:1000]

print(f"Toplam Doküman Sayısı: {len(documents)}")

# 3. Embedding Oluşturma
# FAISS float32 formatındaki NumPy matrisleri ile çalışır!
doc_embeddings = model.encode(documents, convert_to_tensor=False)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# 4. Vektörleri Normalize Etme (Cosine Similarity için Şart)
# Inner Product (IP) kullanarak Cosine Similarity elde etmek için vektör uzunluğunu 1'e eşitlemeliyiz.
faiss.normalize_L2(doc_embeddings)

# 5. FAISS İndeksi Oluşturma
dimension = doc_embeddings.shape[1]  # MiniLM için bu değer 384'tür
index = faiss.IndexFlatIP(dimension) # Inner Product (Kosinüs Benzerliği için)

# Vektörleri FAISS Indeksine Ekleme
index.add(doc_embeddings)
print(f"FAISS İndeksindeki Toplam Vektör: {index.ntotal}")

# 6. Arama (Search) Yapma
query = "Onkoloji ve tümör tedavisi"
query_embedding = model.encode([query], convert_to_tensor=False)
query_embedding = np.array(query_embedding).astype('float32')
faiss.normalize_L2(query_embedding)

# Top-K Arama: En yakın 3 sonucu getir
k = 3
distances, indices = index.search(query_embedding, k)

# 7. Sonuçları Yazdırma
print(f"\nSorgu: '{query}'")
print("="*50)

for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"Derece {rank}: (Skor: {score:.4f})")
    print(f"Doküman ID: {idx}")
    print(f"İçerik: {documents[idx]}\n")