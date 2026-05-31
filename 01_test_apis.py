# 01_test_apis.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import google.generativeai as genai

load_dotenv()

TEST_PROMPT = "Merhaba, bir cümleyle kendini tanıt."

# ── GPT-4o ───────────────────────────────────
print("GPT-4o test ediliyor...")
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": TEST_PROMPT}],
        max_tokens=100
    )
    print("GPT-4o ✓:", response.choices[0].message.content[:80])
except Exception as e:
    print("GPT-4o HATA:", e)

print()

# ── Gemini 1.5 Flash ─────────────────────────
print("Gemini test ediliyor...")
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(TEST_PROMPT)
    print("Gemini ✓:", response.text[:80])
except Exception as e:
    print("Gemini HATA:", e)

print()

# ── Llama 3.1 ────────────────────────────────
print("Llama 3.1 test ediliyor...")
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1", "prompt": TEST_PROMPT, "stream": False},
        timeout=120
    )
    print("Llama 3.1 ✓:", response.json()["response"][:80])
except Exception as e:
    print("Llama 3.1 HATA:", e)

print()

# ── Mistral ──────────────────────────────────
print("Mistral test ediliyor...")
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": TEST_PROMPT, "stream": False},
        timeout=120
    )
    print("Mistral ✓:", response.json()["response"][:80])
except Exception as e:
    print("Mistral HATA:", e)

print()
print("Test tamamlandı!")