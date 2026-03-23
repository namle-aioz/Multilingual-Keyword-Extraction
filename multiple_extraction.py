import csv
import json
import os
import re
import time
from collections import Counter

import faiss
import numpy as np
from langdetect import LangDetectException, detect_langs
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer

try:
    import stopwordsiso
    HAS_STOPWORDS = True
except ImportError:
    HAS_STOPWORDS = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.npy"
DATA_CSV = "data.csv"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

embed_model = SentenceTransformer(MODEL_NAME)

def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def get_dynamic_stopwords(lang_code):
    if not HAS_STOPWORDS:
        return []
    try:
        if stopwordsiso.has_lang(lang_code):
            return list(stopwordsiso.stopwords(lang_code))
        return list(stopwordsiso.stopwords("en"))
    except Exception:
        return list(stopwordsiso.stopwords("en"))

def load_topics_from_csv(filepath):
    topics = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            l1 = row.get("Category", "").strip()
            subcats = [s.strip() for s in row.get("Subcategories", "").split(",")]
            for l2 in subcats:
                if l2:
                    topics.append({"l1": l1, "l2": l2})
    return topics

def get_or_create_index(topics_list):
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        meta = np.load(META_PATH, allow_pickle=True).tolist()
        return index, meta
    
    if not topics_list:
        return None, None

    l2_labels = [f"This text is about {t['l2']}" for t in topics_list]
    embeddings = embed_model.encode(
        l2_labels, 
        batch_size=64, 
        show_progress_bar=False, 
        normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(topics_list, dtype=object))
    return index, topics_list

def detect_lang_with_confidence(text):
    try:
        langs = detect_langs(text)
        if langs:
            return langs[0].lang, round(langs[0].prob, 2)
    except LangDetectException:
        pass
    return "en", 0.0


def normalize_to_english(text):
    cleaned_text = text.strip()
    if not cleaned_text:
        return cleaned_text

    lang, _ = detect_lang_with_confidence(cleaned_text)
    if lang == "en":
        return cleaned_text

    if not HAS_TRANSLATOR:
        return cleaned_text

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(cleaned_text)
        if translated and translated.strip():
            return translated.strip()
    except Exception:
        pass

    return cleaned_text

def extract_ngram_candidates(text, stop_words, ngram_range=(1, 3)):
    clauses = re.split(r'[.,!?()\[\]{}":;\n]+', text.lower())
    
    counts = Counter()
    sw_set = set([s.lower() for s in stop_words]) if stop_words else set()
    
    for clause in clauses:
        words = re.findall(r'\w+', clause)
        if not words:
            continue
        if not words:
            continue
            
        for i in range(len(words)):
            for j in range(ngram_range[0], ngram_range[1] + 1):
                if i + j <= len(words):
                    ngram_tuple = words[i:i+j]
                    
                    if ngram_tuple[0] in sw_set or ngram_tuple[-1] in sw_set:
                        continue
                    
                    ngram_str = " ".join(ngram_tuple)
                    
                    if len(ngram_str) <= 2 or ngram_str.isnumeric():
                        continue
                        
                    counts[ngram_str] += 1
                    
    return counts

def process_multilingual(text, index, meta, embed_model, top_n_kw=10, doc_topic_k=5, keyword_threshold=0.45, word_count=0):
    base_result = {
        "status": "success",
        "word_count": word_count,
        "language_detected": "unknown",
        "language_confidence": 0.0,
        "matched_keywords": []
    }

    if not index or not meta or not text.strip():
        return base_result

    lang, lang_prob = detect_lang_with_confidence(text)
    base_result["language_detected"] = lang
    base_result["language_confidence"] = lang_prob
    
    dynamic_stop_words = get_dynamic_stopwords(lang)

    doc_vec = embed_model.encode([text], normalize_embeddings=True).astype("float32")
    
    doc_distances, doc_indices = index.search(doc_vec, k=doc_topic_k)
    
    valid_topics = []
    valid_topic_labels = []
    
    for i in range(doc_topic_k):
        dist = float(doc_distances[0][i])
        idx = doc_indices[0][i]
        if dist > 0.30: 
            match = meta[idx]
            valid_topics.append(match)
            valid_topic_labels.append(f"This text is about {match['l2']}")

    if not valid_topics:
        return base_result

    valid_topic_vecs = embed_model.encode(valid_topic_labels, normalize_embeddings=True).astype("float32")

    ngram_counts = extract_ngram_candidates(text, dynamic_stop_words, ngram_range=(1, 3))
    candidates = list(ngram_counts.keys())
    
    if not candidates:
        return base_result

    cand_texts = [f"This text is about {c}" for c in candidates]
    cand_vecs = embed_model.encode(cand_texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")

    d = valid_topic_vecs.shape[1]
    topic_index = faiss.IndexFlatIP(d)
    topic_index.add(valid_topic_vecs)

    confidences, topic_indices = topic_index.search(cand_vecs, k=1)

    matched_terms_dict = {}

    for i, kw in enumerate(candidates):
        best_conf = float(confidences[i][0])
        best_topic_idx = int(topic_indices[i][0])
        
        if best_topic_idx != -1 and best_conf >= keyword_threshold:
            match = valid_topics[best_topic_idx]
            
            if kw not in matched_terms_dict or best_conf > matched_terms_dict[kw]["confidence"]:
                matched_terms_dict[kw] = {
                    "l1_topic": match["l1"],
                    "l2_topic": match["l2"],
                    "matched_term": kw,
                    "occurrences": ngram_counts[kw],
                    "confidence": round(best_conf, 4)
                }

    sorted_terms = sorted(list(matched_terms_dict.keys()), key=len, reverse=True)
    unique_terms = []
    for term in sorted_terms:
        if not any(term in other for other in unique_terms if term != other):
            unique_terms.append(term)

    matched_keywords = [matched_terms_dict[t] for t in unique_terms]
    matched_keywords.sort(key=lambda x: (x["confidence"], x["occurrences"]), reverse=True)

    base_result["matched_keywords"] = matched_keywords[:top_n_kw]

    return base_result

if __name__ == "__main__":
    topics_data = load_topics_from_csv(DATA_CSV)
    active_index, active_meta = get_or_create_index(topics_data)

    while True:
        try:
            user_input = input("\nEnter text to analyze (or 'exit' to quit): ")
            word_count = len(user_input.split())
            if word_count == 0 or word_count > 2000:
                print("Please enter text between 1 and 2000 words.")
                continue
            user_input = remove_html(user_input)
            user_input = normalize_to_english(user_input)

            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if not user_input:
                continue

            t_start = time.perf_counter()
            res = process_multilingual(user_input, active_index, active_meta, embed_model, word_count=word_count)
            t_end = time.perf_counter()
            
            print()
            print(json.dumps(res, indent=2, ensure_ascii=False))
            print(f"Time taken: {t_end - t_start:.4f}s")
            
        except KeyboardInterrupt:
            print("\nProgram stopped.")
            break