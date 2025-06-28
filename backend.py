
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import os
import re
import torch
import json
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from difflib import SequenceMatcher
import random
from typing import List
import requests
from pydantic import BaseModel
import numpy as np
import requests
from dotenv import load_dotenv

# === CONFIGURATION ===
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL = "huggingface4_-_zephyr-7b-beta"

nltk.data.path.append("/usr/local/share/nltk_data")
nltk.download('punkt')

# Load environment variables from .env file
load_dotenv()

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://zafirahhh.github.io"],  # Only allow your frontend
    allow_credentials=False,  # Set to False for widest compatibility
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Load Knowledge Base ===
KNOWLEDGE_PATH = os.path.join("data", "nursing_guide_cleaned.txt")
with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
    text = f.read()

def load_chunks_from_text(text, max_len=300):
    paragraphs = re.split(r'\n\s*\n', text)
    grouped = []
    buffer = ""
    for para in paragraphs:
        if re.match(r'^[\u2022\*-]|^\d+\.', para.strip()):
            buffer += para.strip() + " "
        else:
            if buffer:
                grouped.append(buffer.strip())
                buffer = ""
            grouped.append(para.strip())
    if buffer:
        grouped.append(buffer.strip())
    return [p for p in grouped if len(p.split()) > 5]

chunks = load_chunks_from_text(text)

# === Load Model and Chunk Embeddings ===
def get_model():
    cache_dir = os.environ.get("HF_HOME", "/tmp/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

model = get_model()
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

# Main QA Logic
def answer_from_knowledge_base(question: str, return_summary=True):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_chunk = chunks[best_idx]

    question_keywords = set(re.findall(r'\w+', question.lower()))
    candidate_sents = [
        s.strip() for s in sent_tokenize(best_chunk)
        if 6 <= len(s.split()) <= 35 and not any(x in s for x in ['|', ':'])
    ]
    if not candidate_sents:
        return best_chunk

    for sent in candidate_sents:
        words = set(re.findall(r'\w+', sent.lower()))
        if len(words & question_keywords) >= 2:
            return sent

    comma_sents = [s for s in candidate_sents if ',' in s]
    if comma_sents:
        return max(comma_sents, key=lambda s: len(s))

    best_score = 0
    best_sent = candidate_sents[0]
    for sent in candidate_sents:
        sim = SequenceMatcher(None, question.lower(), sent.lower()).ratio()
        if sim > best_score:
            best_score = sim
            best_sent = sent

    return best_sent


# Quiz Generator
def generate_quiz_from_guide(prompt: str):
    # Example basic quiz generation logic
    return (
        "\U0001F9E0 Quiz Time!\n"
        "Q: What is the initial management for a febrile seizure?\n"
        "A. Administer antibiotics\n"
        "B. Cool the child\n"
        "C. Administer paracetamol\n"
        "D. Call ICU\n"
        "\nType the correct option (A/B/C/D)."
    )

# === Request Schema ===
class QueryRequest(BaseModel):
    query: str

class QuizAnswer(BaseModel):
    question: str
    answer: str

class QuizEvaluationRequest(BaseModel):
    responses: List[QuizAnswer]

# === Helper Functions ===
@app.get("/")
def read_root():
    return {"message": "KKH Nursing Chatbot API is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "kkh-nursing-chatbot"}

def clean_paragraph(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        cleaned = re.sub(r"^[\u2022\-\u2013\*\d\.\s]+", "", line).strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    paragraph = " ".join(cleaned_lines)
    paragraph = re.sub(r'\s+', ' ', paragraph).strip()
    return paragraph

def extract_summary_sentences(text: str, max_sentences=3) -> str:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 5]
    formula_lines = [
        line for line in lines
        if re.search(r'70\s*\+\s*\(?.*?age.*?\)?[^\|\n]*', line, re.IGNORECASE)
        or re.search(r'expected systolic bp.*?70.*?age', line, re.IGNORECASE)
    ]
    if formula_lines:
        matches = []
        for line in formula_lines:
            found = re.findall(r'70\s*\+\s*\(?.*?age.*?\)?[^\|\n]*', line, re.IGNORECASE)
            matches.extend(found)
        if matches:
            return '\n'.join(f'- Expected systolic BP formula: {m.strip()}' for m in matches[:max_sentences])
    sentences = [s.strip() for s in sent_tokenize(text) if 10 < len(s.strip()) < 200]
    key_sents = [s for s in sentences if ':' not in s and '|' not in s and len(s.split()) <= 25]
    fallback = key_sents[:max_sentences] or sentences[:max_sentences]
    return '\n'.join(f'- {s}' for s in fallback) if fallback else 'No relevant sentence found.'

def find_best_answer(user_query, chunks, chunk_embeddings, top_k=5):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.topk(k=top_k).indices.tolist()

    question_keywords = set(re.findall(r'\w+', user_query.lower()))
    best_sent = ""
    best_chunk = ""
    sentence_scores = []

    short_prompt = bool(re.search(r'\b(one|two|three|four|five|\d+|signs|types|examples|causes|normal|range|table|bp|hr|rr|rate|values)\b', user_query.lower()))
    fact_query = bool(re.match(r'^(what|which|how|when|who|name|list)\b', user_query.strip().lower()))

    for idx in top_indices:
        chunk = chunks[idx]

        if (
            "© kk women's" in chunk.lower()
            or "the baby bear book" in chunk.lower()
            or "loi v-ter" in chunk.lower()
            or len(chunk.strip()) < 40
            or re.match(r'^(table|figure|section|chapter)\s*\d+|^table|^figure|^section|^chapter', chunk.strip().lower())
            or (len(chunk.split()) < 15 and chunk.strip().endswith('Assessment'))
        ):
            continue

        sentences = [s.strip() for s in sent_tokenize(chunk) if 8 < len(s.split()) < 50]

        for sent in sentences:
            sent_lower = sent.lower()
            sent_keywords = set(re.findall(r'\w+', sent_lower))
            overlap = len(sent_keywords & question_keywords)
            sim = SequenceMatcher(None, user_query.lower(), sent_lower).ratio()
            trigger_score = any(phrase in sent_lower or phrase in user_query.lower() for phrase in [
                "red flag", "warning signs", "danger signs", "signs of deterioration",
                "signs of distress", "sepsis indicators", "critically ill child", "high risk symptoms"
            ])
            score = overlap + sim + (1 if trigger_score else 0)
            sentence_scores.append((score, sent, chunk))

    sentence_scores.sort(reverse=True, key=lambda x: x[0])
    top_sents = [s[1] for s in sentence_scores[:3] if s[0] > 1.5]
    best_sent = top_sents[0] if top_sents else ""
    best_chunk = sentence_scores[0][2] if sentence_scores else ""

    if short_prompt or fact_query:
        sentences = [s.strip() for s in sent_tokenize(best_chunk) if 6 <= len(s.split()) <= 20]
        filtered = [s for s in sentences if any(q in s.lower() for q in question_keywords)]
        final = filtered or sentences[:3]
        return {
            "summary": "\n".join(final),
            "full": "\n".join(final)
        }

    # Contextual fallback if not short/fact
    lines = best_chunk.splitlines()
    result_lines = []
    found = False
    for i, line in enumerate(lines):
        if best_sent in line:
            found = True
            if re.match(r'^[\u2022\-\*\d]+[\.)]?\s', line.strip()):
                start = i
                while start > 0 and re.match(r'^[\u2022\-\*\d]+[\.)]?\s', lines[start - 1].strip()):
                    start -= 1
                end = i
                while end + 1 < len(lines) and re.match(r'^[\u2022\-\*\d]+[\.)]?\s', lines[end + 1].strip()):
                    end += 1
                result_lines = lines[start:end + 1]
            else:
                para = [line]
                j = i - 1
                while j >= 0 and lines[j].strip() and not re.match(r'^[\u2022\-\*\d]+[\.)]?\s', lines[j].strip()):
                    para.insert(0, lines[j])
                    j -= 1
                j = i + 1
                while j < len(lines) and lines[j].strip() and not re.match(r'^[\u2022\-\*\d]+[\.)]?\s', lines[j].strip()):
                    para.append(lines[j])
                    j += 1
                result_lines = para
            break

    summary = '\n'.join([l.strip() for l in result_lines if l.strip()]) if found else '\n'.join(top_sents)

    # === Shortened Answer Formatting ===
    if not summary.strip():
        formula_lines = []

        for idx in top_indices:
            chunk = chunks[idx]

            # Match formulas like "Expected systolic BP"
            matches = re.findall(r'(expected systolic bp.*?(70\s*\+\s*\(?age.*?\)?))', chunk, re.IGNORECASE)
            if not matches:
                matches = re.findall(r'(70\s*\+\s*\(?\s*age.*?\)?)', chunk, re.IGNORECASE)

            for m in matches:
                if isinstance(m, tuple):
                    formula_lines.append(m[0].strip())
                else:
                    formula_lines.append(m.strip())

            # Convert table-like rows to concise sentences
            for line in chunk.splitlines():
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]

                    # Systolic BP formula embedded in a table
                    if len(parts) >= 2 and any("age" in p.lower() for p in parts) and any("70" in p for p in parts):
                        sentence = f"Systolic BP: {parts[1]} mmHg."
                        formula_lines.append(sentence)

                    # Urine output table
                    elif "urine" in user_query.lower() and len(parts) == 2:
                        sentence = f"Urine output: {parts[1]} ml/kg/h."
                        formula_lines.append(sentence)

                    # Vitals table
                    elif len(parts) >= 3:
                        age_group, hr, rr, *bp = parts
                        if "heart rate" in user_query.lower():
                            formula_lines.append(f"HR: {hr} bpm.")
                        if "respiratory" in user_query.lower() or "resp rate" in user_query.lower():
                            formula_lines.append(f"RR: {rr} breaths/min.")
                        if "bp" in user_query.lower() or "blood pressure" in user_query.lower():
                            if bp:
                                formula_lines.append(f"BP: {bp[0]} mmHg.")

        if formula_lines:
            formatted = formula_lines[0]  # Return only the shortest and most relevant formula
            return {"summary": formatted, "full": formatted}

    # Final fallback: concise user-friendly message
    return {
        "summary": summary.split("\n")[0] if summary else "No answer found.",
        "full": clean_paragraph(best_chunk).split("\n")[0] if best_chunk else "No additional information."
    }

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"  # ✅ Missing "Bearer"
}

def generate_with_zephyr(prompt: str):
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.7,
            "max_new_tokens": 512
        }
    }
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"⚠️ Generation failed: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error: {e}"

@app.get("/")
def read_root():
    return {"message": "KKH Nursing Chatbot API is running!"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_query = request.query
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.topk(k=3).indices.tolist()
    context = "\n\n".join([chunks[i] for i in top_indices])

    prompt = f"""
You are a helpful paediatric nurse assistant chatbot. Use the context below to answer the user's question as accurately and logically as possible.

Context:
{context}

Question:
{user_query}

Answer:
"""
    answer = generate_with_zephyr(prompt)
    return {"answer": answer}


@app.post("/search")
async def search(query: QueryRequest):
    result = find_best_answer(query.query, chunks, chunk_embeddings)
    return result

@app.get("/quiz")
def generate_quiz(n: int = 5, topic: str = None):
    filtered_chunks = [c for c in chunks if topic and topic.lower() in c.lower()] if topic else chunks
    quiz = []
    attempts = 0
    max_attempts = n * 10  # Prevent infinite loop
    while len(quiz) < n and attempts < max_attempts:
        chunk = random.choice(filtered_chunks)
        sentences = [s.strip() for s in sent_tokenize(chunk) if 6 <= len(s.split()) <= 25]
        sentences = list(set(sentences))  # Deduplicate
        if len(sentences) < 5:
            attempts += 1
            continue  # Ensure at least 5 unique options
        correct = random.choice(sentences)
        distractors = [s for s in sentences if s != correct]
        selected = random.sample(distractors, 4) + [correct]
        random.shuffle(selected)
        quiz.append({
            "question": ("What is the correct statement regarding " + topic) if topic else "What is the correct statement regarding this medical topic?",
            "options": selected,
            "answer": correct,
            "context": chunk
        })
        attempts += 1
    return {"quiz": quiz}

@app.post("/quiz/evaluate")
def evaluate_quiz(request: QuizEvaluationRequest):
    feedback = []
    for response in request.responses:
        question_text = response.question.lower().strip()
        given_answer = response.answer.strip()

        best_chunk = ""
        best_score = 0
        best_match = ""

        for chunk in chunks:
            for sent in sent_tokenize(chunk):
                score = SequenceMatcher(None, sent.lower(), given_answer.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = sent
                    best_chunk = chunk

        correct = best_match if best_score > 0.7 else "Unknown"
        explanation = extract_summary_sentences(best_chunk, max_sentences=2) if best_chunk else "No explanation found."

        feedback.append({
            "question": response.question,
            "your_answer": given_answer,
            "correctAnswer": correct,
            "correct": given_answer == correct,
            "explanation": explanation
        })

    return feedback

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

