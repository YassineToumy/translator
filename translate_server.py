#!/usr/bin/env python3
"""
NLLB-200 Translation Server
Loads facebook/nllb-200-distilled-1.3B once, serves /translate endpoint.
Deploy on Infomaniak server: uvicorn translate_server:app --host 0.0.0.0 --port 8080
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-1.3B")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

log.info(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
VOCAB = tokenizer.get_vocab()
log.info("Model loaded ✅")

app = FastAPI(title="NLLB Translation Server")


class TranslateRequest(BaseModel):
    text:     str
    src_lang: str = "fra_Latn"   # French
    tgt_lang: str = "arb_Arab"   # Arabic


class TranslateResponse(BaseModel):
    translation: str
    src_lang:    str
    tgt_lang:    str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    text = req.text.strip()[:2000]   # truncate to avoid OOM

    try:
        tokenizer.src_lang = req.src_lang
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        forced_bos = VOCAB.get(req.tgt_lang)
        if forced_bos is None:
            raise ValueError(f"Unknown language code: {req.tgt_lang}")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=MAX_LENGTH,
            )
        translation = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return TranslateResponse(
            translation=translation,
            src_lang=req.src_lang,
            tgt_lang=req.tgt_lang,
        )
    except Exception as e:
        import traceback
        log.error(f"Translation error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)