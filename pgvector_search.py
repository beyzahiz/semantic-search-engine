import json
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# 1. Model Yükleme
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Veriyi Dış Dosyadan Okuma (Data Separation)
with open('documents.json', 'r', encoding='utf-8') as f:
    raw_documents = json.load(f)

print(f"JSON Dosyasından {len(raw_documents)} Adet Doküman Okundu.")

# 3. PostgreSQL Veritabanı Bağlantısı
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="your_password", 
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# pgvector eklentisini veritabanında aktif etme ve psycopg2'ye kaydetme
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
register_vector(conn)

# 4. Tablo Oluşturma
# embedding sütunu VECTOR(384) tipinde tanımlanıyor!
cur.execute("DROP TABLE IF EXISTS documents;")
cur.execute("""
    CREATE TABLE documents (
        id INT PRIMARY KEY,
        title VARCHAR(255),
        content TEXT,
        category VARCHAR(50),
        embedding VECTOR(384)
    );
""")
conn.commit()

# 5. Verileri Vektörleştirip PostgreSQL'e Ekleme (Insertion)
for doc in raw_documents:
    # Metni vektöre çeviriyoruz (384 boyutlu float listesi)
    vector = model.encode(doc['content']).tolist()
    
    # SQL sorgusu ile veritabanına ekleme
    cur.execute(
        """
        INSERT INTO documents (id, title, content, category, embedding)
        VALUES (%s, %s, %s, %s, %s);
        """,
        (doc['id'], doc['title'], doc['content'], doc['category'], vector)
    )

conn.commit()
print("Tüm veriler ve vektörler PostgreSQL veritabanına kaydedildi!")

# 6. pgvector Üzerinden Anlamsal Arama (Semantic Search Query)
query = "Onkoloji ve kanser tedavisindeki gelişmeler"
query_vector = model.encode(query).tolist()

# SQL Sorgusu: <=> operatörü Cosine Distance hesaplar. 
# 1 - (embedding <=> query_vector) ifadesi bize Cosine Similarity verir.
search_sql = """
    SELECT id, title, content, category, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY similarity DESC
    LIMIT 2;
"""

cur.execute(search_sql, (query_vector,))
results = cur.fetchall()

# 7. Sonuçları Yazdırma
print(f"\nSorgu: '{query}'\n" + "="*50)
for row in results:
    doc_id, title, content, category, similarity = row
    print(f"Başlık: {title} | Kategori: {category}")
    print(f"İçerik: {content}")
    print(f"Benzerlik Skoru: {similarity:.4f}\n")

# Bağlantıları Kapatma
cur.close()
conn.close()