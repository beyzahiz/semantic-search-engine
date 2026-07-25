# 🔍 Semantic Search Engine (Anlamsal Arama Motoru)

Bu proje, geleneksel kelime bazlı (Lexical/Keyword) arama yöntemlerinin aksine, metinlerin **anlamsal yakınlıklarını (Contextual Similarity)** analiz ederek arama yapan uçtan uca bir **Semantic Search Engine** mimarisidir.

Projede LLM (Large Language Model) çağrıları yapılmadan önce vektör matematiği, embedding modelleri ve FAISS indeksi kullanılarak milyarlarca veri üzerinde milisaniyeler seviyesinde arama yapabilme altyapısı kurgulanmıştır.

---

## 🛠️ Kullanılan Teknolojiler ve Mimari

* **Python 3.10+**
* **Sentence Transformers (`all-MiniLM-L6-v2`):** Metinleri 384 boyutlu vektör uzayına (Embedding) dönüştürür.
* **FAISS (Facebook AI Similarity Search):** Vektörleri bellek üzerinde indeksler ve ultra hızlı en yakın komşu (Nearest Neighbor) aramasını gerçekleştirir.
* **NumPy & PyTorch:** Vektör normalizasyonu ($L_2$ Normalization) ve matris işlemleri için.
* **JSON Data Separation:** İş mantığı ile veri katmanının birbirinden ayrılması.

---

## 📐 Nasıl Çalışır? (Mimari Akış)

1. **Data Loading:** Metinler dış katman olan `documents.json` dosyasından okunur.
2. **Vectorization (Embedding):** Yüklenen metinler Hugging Face üzerindeki `all-MiniLM-L6-v2` derin öğrenme modeli ile 384 boyutlu vektör dizilerine dönüştürülür.
3. **Normalization:** Kosinüs Benzerliğini (Cosine Similarity) vektörlerin noktasal çarpımıyla (Inner Product) saniyeler içinde hesaplayabilmek için $L_2$ normalizasyonu uygulanır.
4. **FAISS Indexing:** Normalize edilen vektörler `faiss.IndexFlatIP` indeksine yüklenir.
5. **Top-K Search:** Kullanıcının girdiği arama sorgusu vektörleştirilir ve vektör uzayında en yakın $K$ adet doküman milisaniyeler seviyesinde sıralanarak döndürülür.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Repoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADI/semantic-search-engine.git](https://github.com/KULLANICI_ADI/semantic-search-engine.git)
cd semantic-search-engine
```

### 2. Sanal Ortamı Oluşturun ve Aktif Edin
Terminalde proje klasörünün içinde olduğunuzdan emin olun, ardından sanal ortamı (virtual environment) oluşturup aktif edin:

```bash
# Sanal ortamı oluşturun
python3 -m venv venv

# Sanal ortamı aktif edin (macOS / Linux)
source venv/bin/activate

# (Eğer Windows kullanıyorsanız)
# venv\Scripts\activate
```

### 3. Gerekli Kütüphaneleri Yükleyin
```bash
pip install sentence-transformers faiss-cpu numpy
```

### 4. Uygulamayı Çalıştırın
```bash
python main.py
```

---

## 📊 Benchmark ve Performans

Projenin hız ve ölçeklenebilirlik performansını ölçmek amacıyla yapılan test sonuçları aşağıda listelenmiştir:

* **Embedding Modeli:** `all-MiniLM-L6-v2` (384 Boyutlu Vektör Çıktısı)
* **Veri Kümesi:** 1.000 Doküman (Ölçeklenmiş Makale Metinleri)
* **Arama Yöntemi:** $L_2$ Normalizasyonu + Cosine Similarity via Inner Product (`IndexFlatIP`)
* **Vektörleştirme Süresi (1000 Doküman):** ~1.2 - 1.8 saniye (CPU üzerinde)
* **FAISS Arama Süresi (Top-3 Search):** $< 5\text{ ms}$ (Milisaniye seviyesinde gecikme)



