import csv
import json
import os
import re
import time
import warnings
from collections import defaultdict

import faiss
import numpy as np
from langdetect import LangDetectException, detect_langs

from sentence_transformers import SentenceTransformer

try:
    import stopwordsiso
    HAS_STOPWORDS = True
except ImportError:
    HAS_STOPWORDS = False

INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.npy"
DATA_CSV = "data.csv"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

embed_model = SentenceTransformer(MODEL_NAME)

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

    l2_labels = [t["l2"] for t in topics_list]
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

def clean_overlapping_keywords(keywords_list):
    sorted_kws = sorted(keywords_list, key=len, reverse=True)
    unique_kws = []
    for kw in sorted_kws:
        if not any(kw in other for other in unique_kws if kw != other):
            unique_kws.append(kw)
    return unique_kws

def extract_keywords_st(text, stop_words=None, top_n=10, ngram_range=(1, 3)):
    words = text.split()
    if not words:
        return []
    
    sw_set = set([s.lower() for s in stop_words]) if stop_words else set()
    candidates = set()
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i+n]
            if all(w.lower() in sw_set for w in ngram_words):
                continue
            cand = " ".join(ngram_words)
            candidates.add(cand)
            
    candidates = list(candidates)
    if not candidates:
        return []
        
    doc_embedding = embed_model.encode([text], normalize_embeddings=True).astype("float32")
    cand_embeddings = embed_model.encode(candidates, normalize_embeddings=True).astype("float32")
    
    distances = np.dot(cand_embeddings, doc_embedding.T).flatten()
    top_indices = distances.argsort()[::-1][:top_n]
    
    return [(candidates[i], float(distances[i])) for i in top_indices]

def process_multilingual(text, index, meta, top_n_kw=10, threshold=0.45):
    word_count = len(text.split())
    
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
    
    extracted = extract_keywords_st(
        text,
        stop_words=dynamic_stop_words,
        top_n=top_n_kw,
        ngram_range=(1, 3)
    )

    if not extracted:
        return base_result

    kw_strings = [k[0] for k in extracted]
    kw_vecs = embed_model.encode(kw_strings, normalize_embeddings=True).astype("float32")
    
    distances, indices = index.search(kw_vecs, k=1)

    topic_groups = defaultdict(lambda: {
        "matched_terms": set(), "l1": "", "l2": ""
    })

    for i, (kw, _) in enumerate(extracted):
        conf = float(distances[i][0])
        if conf < threshold:
            continue
        
        match = meta[indices[i][0]]
        count = len(re.findall(r"\b{}\b".format(re.escape(kw)), text, re.IGNORECASE))
        
        if count > 0:
            key = (match["l1"], match["l2"])
            group = topic_groups[key]
            group["matched_terms"].add(kw)
            group["l1"] = match["l1"]
            group["l2"] = match["l2"]

    matched_keywords = []
    for v in topic_groups.values():
        clean_terms = clean_overlapping_keywords(list(v["matched_terms"]))
        for term in clean_terms:
            term_count = len(re.findall(r"\b{}\b".format(re.escape(term)), text, re.IGNORECASE))
            if term_count > 0:
                matched_keywords.append({
                    "l1_topic": v["l1"],
                    "l2_topic": v["l2"],
                    "matched_term": term,
                    "occurrences": term_count
                })

    matched_keywords.sort(key=lambda x: x["occurrences"], reverse=True)
    base_result["matched_keywords"] = matched_keywords

    return base_result

if __name__ == "__main__":
    topics_data = load_topics_from_csv(DATA_CSV)
    active_index, active_meta = get_or_create_index(topics_data)

    while True:
        try:
            user_input = input("\n[INPUT TEXT]: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if not user_input:
                continue

            t_start = time.perf_counter()
            res = process_multilingual(user_input, active_index, active_meta)
            t_end = time.perf_counter()
            
            print()
            print(json.dumps(res, indent=2, ensure_ascii=False))
            print(f"Time taken: {t_end - t_start:.4f}s")
            
        except KeyboardInterrupt:
            print("\nProgram stopped.")
            break