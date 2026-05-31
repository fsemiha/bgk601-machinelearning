# BGK601 - LLM Tabanlı Erişim Yönetimi Anomali Tespiti

Bu repository, BGK601 dersi kapsamında hazırlanan **Büyük Dil Modelleri ile Erişim Yönetimi Anomali Tespiti ve Politika Analizi** projesine ait veri seti, değerlendirme kodları, RAG yapısı, ortam dosyaları ve model sonuçlarını içermektedir.

## Proje Başlığı

**Büyük Dil Modelleri ile Erişim Yönetimi Anomali Tespiti ve Politika Analizi**

## Proje Özeti

Bu projede, büyük dil modellerinin kurumsal erişim yönetimi ve IAM odaklı güvenlik senaryolarındaki performansı değerlendirilmiştir.

Benchmark veri seti Türkçe ve sentetik olarak hazırlanmıştır. Senaryolar dört ana kategoriden oluşmaktadır:

- Anomali tespiti
- Tehdit sınıflandırma
- Yetki analizi
- Politika uyumu

Amaç; ticari ve açık kaynaklı büyük dil modellerini doğruluk, gecikme süresi, token kullanımı ve RAG katkısı açısından karşılaştırmaktır.

## Değerlendirilen Modeller

Projede aşağıdaki modeller değerlendirilmiştir:

- Claude
- GPT-4o
- Llama 3.1 8B
- Mistral 7B
- Llama 3.1 8B + RAG

## Veri Seti

Benchmark veri seti toplam **205 sentetik Türkçe erişim yönetimi/güvenlik senaryosundan** oluşmaktadır.

Veri seti gerçek kullanıcı verisi, gerçek kurum logu veya üretim ortamı bilgisi içermemektedir.

Kategori dağılımı:

| Kategori | Örnek Sayısı |
|---|---:|
| Anomali tespiti | 79 |
| Tehdit sınıflandırma | 48 |
| Yetki analizi | 39 |
| Politika uyumu | 39 |
| Toplam | 205 |

Zorluk dağılımı:

| Zorluk Seviyesi | Örnek Sayısı |
|---|---:|
| Kolay | 65 |
| Orta | 78 |
| Zor | 62 |

## Repository Yapısı

```text
bgk601-machinelearning/
│
├── data/
│   └── Benchmark veri seti dosyaları
│
├── results/
│   └── Model değerlendirme çıktıları
│
├── rag_db/
│   └── RAG vektör veritabanı dosyaları
│
├── 01_test_apis.py
│   └── API bağlantı test scripti
│
├── 02_run_benchmark.py
│   └── Tüm benchmark sürecini çalıştıran script
│
├── 03_run_single_model.py
│   └── Tek model değerlendirme scripti
│
├── 04_rag_benchmark.py
│   └── RAG destekli benchmark scripti
│
├── requirements.txt
│   └── Proje için gerekli Python kütüphaneleri
│
├── .env.example
│   └── API anahtarları için örnek ortam değişkenleri dosyası
│
└── README.md
```

## Çalıştırma Ortamı

Bu proje aşağıdaki ortamda geliştirilmiş ve test edilmiştir:

```text
İşletim sistemi: Windows 11
Python sürümü: Python 3.13
Açık kaynak model çalıştırma aracı: Ollama
Yerel modeller: Llama 3.1 8B, Mistral 7B
API tabanlı modeller: GPT-4o, Claude
```

Açık kaynaklı modeller CPU üzerinde çalıştırılmıştır. Bu nedenle Llama 3.1, Mistral ve RAG destekli çalışmalarda gecikme süreleri API tabanlı modellere göre daha yüksek olabilir.

## Kurulum

### 1. Repository'yi klonlama

```bash
git clone https://github.com/fsemiha/bgk601-machinelearning.git
cd bgk601-machinelearning
```

### 2. Sanal ortam oluşturma

Windows için:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux için:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Gerekli kütüphaneleri yükleme

```bash
pip install -r requirements.txt
```

## Ortam Değişkenleri

API tabanlı modelleri çalıştırmak için ortam değişkenleri kullanılmaktadır.

Repository içinde gerçek API anahtarları paylaşılmamıştır. Bunun yerine `.env.example` dosyası örnek olarak eklenmiştir.

Örnek `.env.example` içeriği:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Kullanım için bu dosya kopyalanarak `.env` adıyla oluşturulmalıdır:

```bash
copy .env.example .env
```

macOS/Linux için:

```bash
cp .env.example .env
```

Daha sonra `.env` dosyası içindeki API anahtarları kullanıcı tarafından doldurulmalıdır.

> Not: `.env` dosyası gerçek API anahtarları içerdiği için GitHub'a yüklenmemelidir.

## Ollama ile Yerel Modellerin Hazırlanması

Llama 3.1 ve Mistral modellerini yerel olarak çalıştırmak için Ollama kullanılmaktadır.

Gerekli modeller Ollama üzerinde hazır değilse aşağıdaki komutlarla indirilebilir:

```bash
ollama pull llama3.1
ollama pull mistral
```

Kod içinde kullanılan model isimleri farklıysa, scriptlerdeki model adı ile Ollama üzerindeki model adının aynı olduğundan emin olunmalıdır.

## Kodların Çalıştırılması

### 1. API bağlantılarını test etme

```bash
python 01_test_apis.py
```

Bu script, API anahtarlarının doğru tanımlanıp tanımlanmadığını ve API bağlantılarının çalışıp çalışmadığını test eder.

### 2. Tüm benchmark sürecini çalıştırma

```bash
python 02_run_benchmark.py
```

Bu script, benchmark veri seti üzerinde modellerin değerlendirilmesi için kullanılır.

### 3. Tek model çalıştırma

Aşağıdaki komutlarla tek bir model ayrı ayrı test edilebilir:

```bash
python 03_run_single_model.py gpt4o
python 03_run_single_model.py claude
python 03_run_single_model.py llama
python 03_run_single_model.py mistral
```

### 4. RAG destekli benchmark çalıştırma

```bash
python 04_rag_benchmark.py
```

Bu script, Llama 3.1 modelini RAG desteğiyle çalıştırır ve temel Llama 3.1 sonucu ile karşılaştırma yapılmasını sağlar.

## RAG Yapısı

RAG konfigürasyonunda Llama 3.1 8B modeli MITRE ATT&CK tabanlı güvenlik bilgileriyle desteklenmiştir.

RAG yapısı, ilgili güvenlik bağlamını retrieval adımıyla getirerek model promptuna eklemektedir. Böylece temel Llama 3.1 modeli ile RAG destekli Llama 3.1 modeli karşılaştırılmıştır.

RAG sürecinde genel akış şu şekildedir:

1. Senaryo metni alınır.
2. Senaryo embedding modelinden geçirilir.
3. ChromaDB üzerinde benzer güvenlik doküman parçaları aranır.
4. En ilgili doküman parçaları prompta eklenir.
5. Model güvenlik değerlendirmesi üretir.
6. Çıktı beklenen cevapla karşılaştırılır.

## Değerlendirme Metrikleri

Modeller aşağıdaki tanımlayıcı metrikler üzerinden karşılaştırılmıştır:

- Ortalama doğruluk skoru
- Kategori bazlı başarı
- Zorluk seviyesine göre başarı
- Ortalama gecikme süresi
- Medyan gecikme süresi
- Ortalama token kullanımı
- RAG destekli ve RAG’siz model karşılaştırması

## Temel Bulgular

Projede elde edilen temel bulgular şunlardır:

- Claude en yüksek genel doğruluk skoruna ulaşmıştır.
- GPT-4o en düşük ortalama gecikme süresine sahiptir.
- Llama 3.1 + RAG, temel Llama 3.1 modeline göre daha yüksek doğruluk sağlamıştır.
- RAG doğruluğu artırmış, ancak gecikme süresi ve token kullanımını da yükseltmiştir.
- Modeller en başarılı performansı anomali tespiti kategorisinde göstermiştir.
- Tehdit sınıflandırma ve yetki analizi modeller için daha zorlayıcı olmuştur.

## Sonuç Dosyaları

Model çıktıları `results/` klasöründe tutulmaktadır.

Örnek sonuç dosyaları:

```text
results/
├── gpt4o_results.json
├── claude_results.json
├── llama_results.json
├── llama_rag_results.json
└── mistral_results.json
```

Bu dosyalar, modellerin benchmark senaryolarına verdiği cevapları ve değerlendirme sonuçlarını içermektedir.

## Etik ve Gizlilik Notu

Tüm benchmark senaryoları sentetiktir.

Veri seti aşağıdaki bilgileri içermez:

- Gerçek çalışan verisi
- Gerçek müşteri verisi
- Gerçek kurum logu
- Gerçek IP adresi
- Üretim sistemi bilgisi

Yapay zekâ araçları yalnızca taslak üretimi, kod desteği ve metin düzenleme süreçlerinde yardımcı araç olarak kullanılmıştır. Ground-truth etiketleri araştırmacı tarafından manuel olarak gözden geçirilmiş ve belirlenmiştir.

## Güvenlik Notu

Bu repository içerisinde gerçek API anahtarı, parola, token veya kurumsal gizli bilgi paylaşılmamıştır.

`.env` dosyası yerel kullanım içindir ve GitHub'a yüklenmemelidir. API anahtarları için yalnızca `.env.example` dosyası paylaşılmıştır.


