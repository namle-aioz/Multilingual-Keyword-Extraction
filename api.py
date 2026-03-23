from fastapi import FastAPI
from pydantic import BaseModel
from multiple_extraction import (
    load_topics_from_csv,
    get_or_create_index,
    process_multilingual,
    remove_html,
    DATA_CSV
)

app = FastAPI(title="Multilingual Keyword Extraction API", version="1.0.0")

topics_data = load_topics_from_csv(DATA_CSV)
active_index, active_meta = get_or_create_index(topics_data)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(MODEL_NAME)


class TextInput(BaseModel):
    text: str


class AnalysisResponse(BaseModel):
    status: str
    word_count: int
    language_detected: str
    language_confidence: float
    matched_keywords: list


@app.get("/")
def read_root():
    return {
        "message": "Multilingual Keyword Extraction API",
        "endpoint": "/analyze",
        "method": "POST",
        "description": "Send text to analyze and extract multilingual keywords"
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_text(input_data: TextInput):
    text = remove_html(input_data.text)
    
    if not text.strip():
        return {
            "status": "error",
            "word_count": 0,
            "language_detected": "unknown",
            "language_confidence": 0.0,
            "matched_keywords": []
        }
    
    result = process_multilingual(text, active_index, active_meta, embed_model)
    return result


@app.post("/batch")
def analyze_batch(inputs: list[TextInput]):
    results = []
    for input_data in inputs:
        text = remove_html(input_data.text)
        result = process_multilingual(text, active_index, active_meta, embed_model)
        results.append(result)
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
