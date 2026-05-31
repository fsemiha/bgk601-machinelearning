"""
02_run_benchmark.py
Erişim Yönetimi LLM Benchmark - Final Deney Scripti
BGK601 - Bahar 2025-2026

Kullanım: python 02_run_benchmark.py
"""

import os
import json
import time
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import requests

load_dotenv()

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
BENCHMARK_PATH = "data/benchmark.json"
RESULTS_DIR    = "results"
N_RUNS         = 3
PILOT_LIMIT    = None   # None = tüm dataset, 30 = pilot mod
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

# ─────────────────────────────────────────────
# MODEL İSTEMCİLERİ
# ─────────────────────────────────────────────

def ask_gpt4o(prompt: str) -> tuple[str, float, int]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=300
    )
    latency = (time.time() - t0) * 1000
    text    = response.choices[0].message.content
    tokens  = response.usage.total_tokens
    return text, latency, tokens


def ask_claude(prompt: str) -> tuple[str, float, int]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    t0 = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - t0) * 1000
    text    = response.content[0].text
    tokens  = response.usage.input_tokens + response.usage.output_tokens
    return text, latency, tokens


def ask_ollama(model_name: str, prompt: str) -> tuple[str, float, int]:
    full_prompt = f"{SYSTEM_PROMPT}\n\nSenaryo:\n{prompt}"
    t0 = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":       model_name,
            "prompt":      full_prompt,
            "stream":      False,
            "temperature": TEMPERATURE,
        },
        timeout=300
    )
    latency = (time.time() - t0) * 1000
    text    = response.json()["response"]
    tokens  = response.json().get("eval_count", 0) + response.json().get("prompt_eval_count", 0)
    return text, latency, tokens


def ask_llama31(prompt: str) -> tuple[str, float, int]:
    return ask_ollama("llama3.1", prompt)


def ask_mistral(prompt: str) -> tuple[str, float, int]:
    return ask_ollama("mistral", prompt)


# ─────────────────────────────────────────────
# SKOR HESAPLAMA
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────

def run_benchmark():
    print("=" * 60)
    print("BGK601 - Erişim Yönetimi LLM Benchmark (Final)")
    print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    if PILOT_LIMIT:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for item in dataset:
            by_cat[item["kategori"]].append(item)
        pilot = []
        per_cat = PILOT_LIMIT // len(by_cat)
        for cat, items in by_cat.items():
            pilot.extend(items[:per_cat])
        remaining = PILOT_LIMIT - len(pilot)
        if remaining > 0:
            used_ids = {i["id"] for i in pilot}
            extras   = [i for i in dataset if i["id"] not in used_ids]
            pilot.extend(extras[:remaining])
        dataset = pilot
        print(f"Pilot mod: {len(dataset)} örnek\n")
    else:
        print(f"Tam dataset: {len(dataset)} örnek\n")

    models = {
        "GPT-4o":   ask_gpt4o,
        "Claude":   ask_claude,
        "Llama3.1": ask_llama31,
        "Mistral":  ask_mistral,
    }

    all_results = []

    for model_name, model_fn in models.items():
        print(f"\n{'─'*40}")
        print(f"Model: {model_name}")
        print(f"{'─'*40}")

        model_results = []
        total_score   = 0
        total_latency = 0
        total_tokens  = 0
        errors        = 0

        for i, example in enumerate(dataset):
            run_scores    = []
            run_latencies = []
            run_tokens    = []
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

                # Rate limit koruması
                if model_name in ["GPT-4o", "Claude"]:
                    time.sleep(0.5)

            avg_score   = sum(run_scores)   / N_RUNS
            avg_latency = sum(run_latencies) / N_RUNS
            avg_tokens  = sum(run_tokens)   / N_RUNS

            total_score   += avg_score
            total_latency += avg_latency
            total_tokens  += avg_tokens

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
            model_results.append(result)
            all_results.append(result)

            if (i + 1) % 10 == 0:
                running_acc = total_score / (i + 1)
                print(f"  [{i+1}/{len(dataset)}] Doğruluk: {running_acc:.1%} | "
                      f"Ort. Gecikme: {total_latency/(i+1):.0f}ms")

        # Model özeti
        n = len(dataset)
        print(f"\n  ✓ {model_name} Tamamlandı:")
        print(f"    Doğruluk:     {total_score/n:.1%}")
        print(f"    Ort. Gecikme: {total_latency/n:.0f} ms")
        print(f"    Ort. Token:   {total_tokens/n:.0f}")
        print(f"    Hata Sayısı:  {errors}")

        from collections import defaultdict
        cat_scores  = defaultdict(list)
        diff_scores = defaultdict(list)
        for r in model_results:
            cat_scores[r["kategori"]].append(r["avg_score"])
            diff_scores[r["zorluk"]].append(r["avg_score"])

        print(f"\n    Kategori Bazlı:")
        for cat, scores in sorted(cat_scores.items()):
            print(f"      {cat:<25} {sum(scores)/len(scores):.1%}")

        print(f"\n    Zorluk Bazlı:")
        for d, scores in sorted(diff_scores.items()):
            print(f"      {d:<10} {sum(scores)/len(scores):.1%}")

    # Sonuçları kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = f"{RESULTS_DIR}/final_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    csv_path = f"{RESULTS_DIR}/final_results_{timestamp}.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"Sonuçlar kaydedildi:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()