from sentence_transformers import SentenceTransformer, util

# 1. Modeli Yükleme
# Hugging Face üzerindeki hafif, hızlı ve oldukça başarılı bir embedding modeli.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Veri Seti (Dokümanlarımız)
documents = [
    "Onkoloji alanında yeni immunoterapi yöntemleri geliştiriliyor.",
    "Yapay zeka modelleri doğal dil işleme alanında devrim yarattı.",
    "Kanser hastaları için yeni tedavi protokolleri onaylandı.",
    "Fenerbahçe dün akşam oynanan futbol maçını 2-1 kazandı."
]

# 3. Sorgu (Kullanıcının Arama Metni)
query = "Kanser tedavisi"

# 4. Metinleri Vektörlere (Embedding) Dönüştürme
doc_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

# 5. Kosinüs Benzerliği Hesaplama
cosine_scores = util.cos_sim(query_embedding, doc_embeddings)

# 6. Sonuçları Eşleştirme ve Yazdırma
print(f"Sorgu: '{query}'\n" + "-"*40)

for i, doc in enumerate(documents):
    score = cosine_scores[0][i].item()
    print(f"Doküman: {doc}")
    print(f"Benzerlik Skoru: {score:.4f}\n")