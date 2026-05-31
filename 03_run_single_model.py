"""
03_run_single_model.py
Tek model çalıştırır - kaldığı yerden devam edebilir
BGK601 - Bahar 2025-2026

Kullanım: 
  python 03_run_single_model.py gpt4o
  python 03_run_single_model.py claude
  python 03_run_single_model.py llama
  python 03_run_single_model.py mistral
"""

import os
import json
import time
import csv
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import requests

load_dotenv()

BENCHMARK_PATH = "data/benchmark.json"
RESULTS_DIR    = "results"
N_RUNS         = 3
TEMPERATURE    = 0.0

SYSTEM_PROMPT = """Sen bir siber güvenlik uzmanısın. 
Sana verilen erişim yönetimi senaryolarını analiz ederek kısa ve net bir değerlendirme yapacaksın.

Her senaryoda şunları belirt:
1. Karar: (tek satırda ana kararın, örn: "ANOMALİ", "NORMAL", "POLİTİKA İHLALİ" vb.)
2. Risk Seviyesi: DÜŞÜK / ORTA / YÜKSEK / KRİTİK
3. Gerekçe: 1-2 cümle açıklama
4. Önerilen Aksiyon: Ne yapılmalı?

Yanıtını her zaman bu yapıda ver."""

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model fonksiyonları ───────────────────────

def ask_gpt4o(prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    t0 = time.time()
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        temperature=TEMPERATURE, max_tokens=300
    )
    return r.choices[0].message.content, (time.time()-t0)*1000, r.usage.total_tokens

def ask_claude(prompt):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    t0 = time.time()
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user","content":prompt}]
    )
    return r.content[0].text, (time.time()-t0)*1000, r.usage.input_tokens+r.usage.output_tokens

def ask_ollama(model_name, prompt):
    full = f"{SYSTEM_PROMPT}\n\nSenaryo:\n{prompt}"
    t0 = time.time()
    r = requests.post("http://localhost:11434/api/generate",
        json={"model":model_name,"prompt":full,"stream":False,"temperature":TEMPERATURE},
        timeout=300)
    data = r.json()
    return data["response"], (time.time()-t0)*1000, data.get("eval_count",0)+data.get("prompt_eval_count",0)

def compute_score(predicted, expected):
    expected_clean  = expected.upper().strip()
    predicted_upper = predicted.upper()
    key = expected_clean.split("/")[0].split(",")[0].strip()
    if key in predicted_upper:
        return 1.0
    mapping = {
        "ANOMALİ":          ["ANOMAL","SUSPICIOUS","ŞÜPHEL"],
        "NORMAL":           ["NORMAL","BEKLENEN","RUTIN"],
        "AŞIRI YETKİ":      ["AŞIRI","FAZLA YETKİ","EXCESS","OVERPRIVILEGE"],
        "UYGUN YETKİ":      ["UYGUN","APPROPRIATE","LEAST PRIVILEGE"],
        "POLİTİKA İHLALİ":  ["İHLAL","VIOLATION","UYUMSUZ"],
        "POLİTİKAYA UYGUN": ["UYGUN","COMPLIANT","KARŞILANMIŞ"],
    }
    for alt in mapping.get(key, []):
        if alt in predicted_upper:
            return 1.0
    return 0.0

# ── Ana fonksiyon ─────────────────────────────

def run_model(model_key):
    models = {
        "gpt4o":   ("GPT-4o",   ask_gpt4o,   0.5),
        "claude":  ("Claude",   ask_claude,   0.5),
        "llama":   ("Llama3.1", lambda p: ask_ollama("llama3.1", p), 0),
        "mistral": ("Mistral",  lambda p: ask_ollama("mistral", p),  0),
    }

    if model_key not in models:
        print(f"Geçersiz model. Seçenekler: {list(models.keys())}")
        sys.exit(1)

    model_name, model_fn, sleep_time = models[model_key]

    # Daha önce tamamlanmış sonuçları yükle
    output_file = f"{RESULTS_DIR}/{model_key}_results.json"
    done_ids = set()
    results = []

    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Kaldığı yerden devam: {len(done_ids)} örnek zaten tamamlandı")

    # Dataset yükle
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    # Tamamlanmayanları filtrele
    remaining = [d for d in dataset if d["id"] not in done_ids]
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Kalan: {len(remaining)}/{len(dataset)} örnek")
    print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    total_score = 0
    errors = 0

    for i, example in enumerate(remaining):
        run_scores = []
        run_latencies = []
        run_tokens = []
        last_response = ""

        for run in range(N_RUNS):
            try:
                response, latency, tokens = model_fn(example["girdi"])
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

            if sleep_time > 0:
                time.sleep(sleep_time)

        avg_score   = sum(run_scores) / N_RUNS
        avg_latency = sum(run_latencies) / N_RUNS
        avg_tokens  = sum(run_tokens) / N_RUNS
        total_score += avg_score

        result = {
            "model":          model_name,
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

        # Her 5 örnekte bir kaydet (crash koruması)
        if (i + 1) % 5 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            completed = len(done_ids) + i + 1
            running_acc = sum(r["avg_score"] for r in results) / len(results)
            print(f"  [{completed}/{len(dataset)}] Doğruluk: {running_acc:.1%} | "
                  f"Ort. Gecikme: {avg_latency:.0f}ms | Kaydedildi ✓")

    # Final kayıt
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Özet
    from collections import defaultdict
    cat_scores  = defaultdict(list)
    diff_scores = defaultdict(list)
    for r in results:
        cat_scores[r["kategori"]].append(r["avg_score"])
        diff_scores[r["zorluk"]].append(r["avg_score"])

    total = len(results)
    genel = sum(r["avg_score"] for r in results) / total

    print(f"\n{'='*50}")
    print(f"✓ {model_name} TAMAMLANDI")
    print(f"  Doğruluk:     {genel:.1%}")
    print(f"  Toplam örnek: {total}")
    print(f"  Hata sayısı:  {errors}")
    print(f"\n  Kategori Bazlı:")
    for cat, scores in sorted(cat_scores.items()):
        print(f"    {cat:<25} {sum(scores)/len(scores):.1%}")
    print(f"\n  Zorluk Bazlı:")
    for d, scores in sorted(diff_scores.items()):
        print(f"    {d:<10} {sum(scores)/len(scores):.1%}")
    print(f"\n  Sonuç: {output_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python 03_run_single_model.py [gpt4o|claude|llama|mistral]")
        sys.exit(1)
    run_model(sys.argv[1].lower())