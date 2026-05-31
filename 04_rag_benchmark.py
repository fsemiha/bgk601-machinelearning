"""
04_rag_benchmark.py
RAG Sistemi - Llama 3.1 + MITRE ATT&CK Bilgi Tabanı
BGK601 - Bahar 2025-2026

Kullanım: python 04_rag_benchmark.py
"""

import os
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

BENCHMARK_PATH = "data/benchmark.json"
RESULTS_DIR    = "results"
N_RUNS         = 1
TEMPERATURE    = 0.0

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("rag_db", exist_ok=True)

# ── MITRE ATT&CK Bilgi Tabanı ─────────────────

MITRE_DOCUMENTS = [
    """T1078 - Valid Accounts (Geçerli Hesapların Kötüye Kullanımı)
Taktik: Initial Access, Persistence, Privilege Escalation
Açıklama: Saldırganlar mevcut hesap kimlik bilgilerini kullanarak sisteme erişir. Ayrılan çalışan hesapları, varsayılan hesaplar veya çalınan kimlik bilgileri kullanılabilir.
Göstergeler: Olağandışı saatlerde giriş, coğrafi olarak imkansız erişimler, devre dışı hesap aktivitesi, çoklu başarısız giriş sonrası başarılı giriş, zombie account erişimi.
Önlemler: MFA zorunluluğu, hesap devre dışı bırakma politikası, düzenli erişim gözden geçirmesi, offboarding prosedürü.""",

    """T1110 - Brute Force (Kaba Kuvvet Saldırısı)
Taktik: Credential Access
Açıklama: Saldırganlar çok sayıda kullanıcı adı ve şifre kombinasyonu dener. Password spraying, credential stuffing ve dictionary attack alt kategorilerdir.
Göstergeler: Kısa sürede çok sayıda başarısız giriş, farklı IP'lerden sistematik denemeler, alfabetik kullanıcı adı denemeleri, 5 dakikada 500 başarısız giriş.
Önlemler: Hesap kilitleme, MFA, rate limiting, IP engelleme.""",

    """T1098 - Account Manipulation (Hesap Manipülasyonu)
Taktik: Persistence, Privilege Escalation
Açıklama: Saldırganlar mevcut hesapların özelliklerini değiştirerek erişimi sürdürür. Gruplara hesap ekleme, izin değiştirme, MFA devre dışı bırakma bu kapsamdadır.
Göstergeler: Kullanıcının kendi hesabını admin grubuna eklemesi, beklenmedik yetki değişiklikleri, MFA devre dışı bırakma, yönetici grubuna bilinmeyen hesap eklenmesi.
Önlemler: Hesap değişiklik logları izleme, yetki değişikliklerinde onay mekanizması.""",

    """T1136 - Create Account (Hesap Oluşturma)
Taktik: Persistence
Açıklama: Saldırganlar kalıcı erişim için yeni hesaplar oluşturur. Gece yarısı admin hesabı oluşturma ve log silme ile birlikte görülmesi APT işaretidir.
Göstergeler: Yetki dışı yeni hesap oluşturma, gece saatlerinde admin hesabı oluşturma, log silmesi ile birlikte hesap oluşturma.
Önlemler: Hesap oluşturma logları izleme, yetki dışı hesap oluşturma alarmları.""",

    """T1530 - Data from Cloud Storage (Bulut Depolamadan Veri Toplama)
Taktik: Collection, Exfiltration
Açıklama: Saldırganlar bulut depolama servislerindeki verilere erişir. Anormal veri indirme hacmi insider threat veya hesap ele geçirme işaretidir.
Göstergeler: Anormal miktarda veri indirme, yeni bulut depolama erişimi, mesai dışı yoğun veri transferi, 50GB veri tek oturumda indirme.
Önlemler: Bulut erişim logları izleme, DLP politikaları, erişim kısıtlamaları.""",

    """T1048 - Exfiltration Over Alternative Protocol (DNS Tünelleme)
Taktik: Exfiltration
Açıklama: Saldırganlar verileri DNS sorguları içine gizleyerek sızdırır. Anlamsız alt alan adlarına çok sayıda küçük DNS sorgusu bu tekniğin işaretidir.
Göstergeler: Anormal DNS sorgu hacmi, anlamsız alt alan adları, küçük boyutlu çok sayıda DNS paketi, her sorguda gömülü veri.
Önlemler: DNS trafiği izleme, anormal DNS sorgularına alarm.""",

    """T1566 - Phishing (Oltalama)
Taktik: Initial Access
Açıklama: Saldırganlar sahte e-posta ile kullanıcıları kandırır. CEO dolandırıcılığı (BEC) ve spear phishing kurumsal ortamlarda yaygındır.
Göstergeler: Sahte alan adları (bir harf fark), acil para transferi talepleri, HR portalı güncelleme bahanesi, kargo takip linki.
Önlemler: E-posta filtreleme, güvenlik farkındalık eğitimi, MFA, DMARC/SPF/DKIM.""",

    """T1486 - Ransomware (Fidye Yazılımı)
Taktik: Impact
Açıklama: Saldırganlar dosyaları şifreleyerek fidye talep eder. Ağ paylaşımlarına yayılma ve CPU artışı erken belirtilerdir.
Göstergeler: Toplu dosya şifreleme, fidye notu, CPU/disk kullanımında ani artış, ağ paylaşımlarına erişim, EDR alarmı.
Önlemler: Düzenli yedekleme, ağ segmentasyonu, EDR kullanımı.""",

    """T1557 - Man-in-the-Middle (Ortadaki Adam)
Taktik: Credential Access, Collection
Açıklama: Saldırganlar ağ trafiğini iki taraf arasına girerek dinler. TLS sertifikası değiştirme girişimi bu saldırının işaretidir.
Göstergeler: Sertifika uyarıları, beklenmedik ARP değişiklikleri, TLS sertifikası manipülasyonu.
Önlemler: TLS/SSL zorunluluğu, sertifika sabitleme, ağ segmentasyonu.""",

    """T1190 - Exploit Public Application (Web Uygulama Saldırısı)
Taktik: Initial Access
Açıklama: Saldırganlar web uygulamalarındaki açıkları kullanır. SQL injection parametreleri (OR 1=1, UNION SELECT) bu saldırının göstergesidir.
Göstergeler: SQL sorgusu içeren HTTP parametreleri, anormal yanıt kodları, beklenmedik sistem komutları.
Önlemler: WAF, güvenli kod geliştirme, düzenli yama.""",

    """Privilege Escalation - Yetki Yükseltme
Açıklama: Kullanıcı kendi rolünün üzerinde bir sisteme erişmeye çalışır. Stajyerin /etc/passwd dosyasına yazma girişimi, HR uzmanının admin grubuna kendini eklemesi buna örnektir.
Göstergeler: Rol dışı sistem erişimi, sudo başarısız deneme, admin paneline yetkisiz erişim, servis hesabının beklenmedik işlem yapması.
IAM Prensibi: Minimum yetki (least privilege), görev ayrılığı (segregation of duties).
Önlemler: RBAC, düzenli yetki gözden geçirme, PAM çözümleri.""",

    """Insider Threat - İçeriden Tehdit
Açıklama: Mevcut veya eski çalışan sisteme zarar verir. Düşük-yavaş veri sızdırma (low-and-slow), işten ayrılmadan önce veri kopyalama yaygın senaryolardır.
Göstergeler: İşten ayrılmadan önce yoğun veri indirme, koordineli çoklu hesap veri dışa aktarımı, uzun süre küçük miktarda veri indirme (6 ay boyunca 15000 kayıt), aktivite düşüşü.
Önlemler: Kullanıcı davranış analizi (UEBA), offboarding prosedürü, veri sınıflandırma.""",

    """Impossible Travel - Fiziksel Olarak İmkansız Konum Değişikliği
Açıklama: Kullanıcı kısa sürede iki farklı coğrafi konumdan giriş yapar. İstanbul'dan 10:00'da ve Londra'dan 12:00'da giriş (uçuş süresi 4 saat) buna örnektir.
Göstergeler: 2 saat arayla farklı kıtalarda giriş, birden fazla ülkeden eş zamanlı oturum.
Risk: Kimlik bilgisi çalınmış olabilir, hesap ele geçirilmiş olabilir.
Önlemler: Coğrafi kısıtlama politikası, impossible travel alarmı, MFA.""",

    """Supply Chain Attack - Tedarik Zinciri Saldırısı
Açıklama: Saldırganlar güvenilir yazılım güncellemelerine zararlı kod ekler. SolarWinds 2020 ve Colonial Pipeline 2021 gerçek dünya örnekleridir.
Göstergeler: Dijital imzalı güncelleme içinde zararlı kod, C2 bağlantısı açan güncelleme, 6 ay boyunca fark edilmeyen backdoor.
Önlemler: Yazılım güncelleme doğrulama, tedarikçi güvenlik değerlendirmesi, ağ trafiği izleme.""",

    """MFA Fatigue - MFA Yorgunluk Saldırısı
Açıklama: Saldırgan çok sayıda MFA bildirimi göndererek kullanıcıyı onaylamaya zorlar. Uber 2022 saldırısı bu teknikle gerçekleşti.
Göstergeler: Kısa sürede onlarca MFA push bildirimi, kullanıcının yorgunluktan onay vermesi.
Risk: MFA teknik olarak çalışıyor ama sosyal mühendislikle aşılıyor. Politika yetersiz.
Önlemler: Number matching MFA, FIDO2/WebAuthn, phishing-resistant MFA.""",

    """Zero Trust - Sıfır Güven Mimarisi
Açıklama: Hiçbir kullanıcı veya cihaza otomatik güven verilmez. Her erişimde doğrulama yapılır. Ağ konumuna dayalı güven kaldırılır.
Prensipler: Her erişimde kimlik doğrulama, minimum yetki, sürekli izleme, mikro segmentasyon.
IAM Bağlantısı: RBAC, ABAC, PAM, JIT (Just-in-Time) erişim.""",

    """SOD - Segregation of Duties (Görev Ayrılığı)
Açıklama: Kritik işlemlerin tek kişide toplanmaması prensibi. Fatura oluşturma ve ödeme onaylama aynı kişide olmamalıdır.
Örnekler: Muhasebecinin hem fatura oluşturması hem ödeme onaylaması SOD ihlali. Geliştiricinin hem kod yazması hem production'a deploy etmesi SOD ihlali.
Önlemler: Rol bazlı erişim kontrolü, dört göz prensibi, otomatik SOD analizi.""",

    """Least Privilege - Minimum Yetki Prensibi
Açıklama: Kullanıcı ve sistemlere yalnızca görevleri için gerekli minimum yetki verilmesi.
İhlal Örnekleri: Stajyerin production sunucularına root erişimi, satış temsilcisinin müşteri verilerini silme yetkisi, log toplama servisinin IAM tam yetkisi.
Orphaned Permissions: Proje bitince iptal edilmeyen geçici yetkiler, departman değişikliğinde temizlenmeyen eski yetkiler.
Önlemler: Düzenli yetki gözden geçirme, otomatik yetki sona erdirme, PAM.""",

    """SIEM Anomali - Güvenlik Bilgi ve Olay Yönetimi
Açıklama: SIEM sisteminin anormal davranışları. 24 saatte hiç alarm üretmemesi kritik bir anomalidir — sessizlik tehlike işareti olabilir.
Göstergeler: Sıfır alarm (log kaynağı kesilmiş olabilir), log manipülasyonu, auth sunucusunda sıfır giriş isteği.
Önlemler: SIEM sağlık izleme, log kaynağı kesintisi alarmı, baseline davranış analizi.""",

    """Credential Stuffing - Kimlik Bilgisi Doldurma
Açıklama: Saldırgan sızdırılmış kullanıcı adı/şifre listesini otomatik olarak dener. Dark web'de satılan kimlik bilgileri bu saldırıda kullanılır.
Göstergeler: Farklı IP'lerden sistematik giriş denemeleri, 10 farklı hesap için şifre sıfırlama, dark web'de şirket kimlik bilgisi tespiti.
Önlemler: MFA, şifre sızdırılma kontrolü (HaveIBeenPwned), rate limiting.""",
]

def build_knowledge_base():
    """MITRE ATT&CK bilgi tabanını ChromaDB'ye yükler."""
    print("Bilgi tabanı oluşturuluyor...")

    embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = []
    for doc in MITRE_DOCUMENTS:
        chunks = text_splitter.split_text(doc)
        docs.extend(chunks)

    vectorstore = Chroma.from_texts(
        texts=docs,
        embedding=embeddings,
        persist_directory="rag_db"
    )

    print(f"Bilgi tabanı hazır: {len(docs)} chunk yüklendi")
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 3) -> str:
    """Sorguya en yakın k dokümanı getirir."""
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


def ask_llama_rag(prompt: str, vectorstore) -> tuple[str, float, int]:
    """Llama 3.1'e RAG ile sorar."""
    context = retrieve_context(vectorstore, prompt)

    rag_prompt = f"""Sen bir siber güvenlik uzmanısın.
Aşağıdaki MITRE ATT&CK ve güvenlik bilgilerini referans alarak erişim yönetimi senaryosunu değerlendir:

--- İLGİLİ GÜVENLİK BİLGİSİ ---
{context}
--- BİTİŞ ---

Senaryo:
{prompt}

Her senaryoda şunları belirt:
1. Karar: (ANOMALİ / NORMAL / POLİTİKA İHLALİ vb.)
2. Risk Seviyesi: DÜŞÜK / ORTA / YÜKSEK / KRİTİK
3. Gerekçe: 1-2 cümle açıklama
4. Önerilen Aksiyon: Ne yapılmalı?"""

    t0 = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":       "llama3.1",
            "prompt":      rag_prompt,
            "stream":      False,
            "temperature": TEMPERATURE,
        },
        timeout=300
    )
    latency = (time.time() - t0) * 1000
    data    = response.json()
    text    = data["response"]
    tokens  = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
    return text, latency, tokens


def compute_score(predicted: str, expected: str) -> float:
    expected_clean  = expected.upper().strip()
    predicted_upper = predicted.upper()
    key = expected_clean.split("/")[0].split(",")[0].strip()
    if key in predicted_upper:
        return 1.0
    mapping = {
        "ANOMALİ":          ["ANOMAL", "SUSPICIOUS", "ŞÜPHEL"],
        "NORMAL":           ["NORMAL", "BEKLENEN", "RUTIN"],
        "AŞIRI YETKİ":      ["AŞIRI", "FAZLA YETKİ", "EXCESS", "OVERPRIVILEGE"],
        "UYGUN YETKİ":      ["UYGUN", "APPROPRIATE", "LEAST PRIVILEGE"],
        "POLİTİKA İHLALİ":  ["İHLAL", "VIOLATION", "UYUMSUZ"],
        "POLİTİKAYA UYGUN": ["UYGUN", "COMPLIANT", "KARŞILANMIŞ"],
    }
    for alt in mapping.get(key, []):
        if alt in predicted_upper:
            return 1.0
    return 0.0


def run_rag_benchmark():
    print("=" * 60)
    print("BGK601 - RAG Benchmark (Llama3.1 + MITRE ATT&CK)")
    print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Bilgi tabanını kur
    vectorstore = build_knowledge_base()

    # Dataset yükle
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nToplam: {len(dataset)} örnek\n")

    # Kaldığı yerden devam
    output_file = f"{RESULTS_DIR}/llama_rag_results.json"
    done_ids = set()
    results = []

    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Kaldığı yerden devam: {len(done_ids)} örnek tamamlandı\n")

    remaining = [d for d in dataset if d["id"] not in done_ids]

    total_score = 0
    errors = 0

    for i, example in enumerate(remaining):
        run_scores    = []
        run_latencies = []
        run_tokens    = []
        last_response = ""

        for run in range(N_RUNS):
            try:
                response, latency, tokens = ask_llama_rag(example["girdi"], vectorstore)
                score = compute_score(response, example["beklenen_cevap"])
                run_scores.append(score)
                run_latencies.append(latency)
                run_tokens.append(tokens)
                last_response = response
            except Exception as e:
                print(f"  HATA (id={example['id']}, run={run+1}): {e}")
                errors += 1
                run_scores.append(0)
                run_latencies.append(0)
                run_tokens.append(0)

        avg_score   = sum(run_scores) / N_RUNS
        avg_latency = sum(run_latencies) / N_RUNS
        avg_tokens  = sum(run_tokens) / N_RUNS
        total_score += avg_score

        result = {
            "model":          "Llama3.1+RAG",
            "id":             example["id"],
            "kategori":       example["kategori"],
            "zorluk":         example["zorluk"],
            "girdi":          example["girdi"][:100] + "...",
            "beklenen":       example["beklenen_cevap"],
            "tahmin_ozet":    last_response[:150] + "...",
            "avg_score":      round(avg_score, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_tokens":     round(avg_tokens, 1),
            "run_scores":     run_scores,
        }
        results.append(result)

        if (i + 1) % 5 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            completed = len(done_ids) + i + 1
            running_acc = sum(r["avg_score"] for r in results) / len(results)
            print(f"  [{completed}/{len(dataset)}] Doğruluk: {running_acc:.1%} | "
                  f"Gecikme: {avg_latency:.0f}ms | Kaydedildi ✓")

    # Final kayıt
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from collections import defaultdict
    cat_scores = defaultdict(list)
    for r in results:
        cat_scores[r["kategori"]].append(r["avg_score"])

    total = len(results)
    genel = sum(r["avg_score"] for r in results) / total

    print(f"\n{'='*60}")
    print(f"✓ Llama3.1+RAG TAMAMLANDI")
    print(f"  Doğruluk:     {genel:.1%}")
    print(f"  Toplam örnek: {total}")
    print(f"  Hata sayısı:  {errors}")
    print(f"\n  Kategori Bazlı:")
    for cat, scores in sorted(cat_scores.items()):
        print(f"    {cat:<25} {sum(scores)/len(scores):.1%}")
    print(f"\n  Sonuç: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_rag_benchmark()