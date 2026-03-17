#!/usr/bin/env python3
"""
Universal Translator — announcements → announcement_translations (PostgreSQL)

Strategy:
  1. Detect source language (en / fr / ar / es)
  2. Translate to English first (pivot), unless already English
  3. From English, translate to all remaining locales
  4. Store one row per locale in announcement_translations

Usage:
    python translator.py              # continuous loop (default)
    python translator.py --once       # single pass then exit
    python translator.py --limit 20   # process max 20 announcements
"""

import argparse
import json
import logging
import os
import re
import time

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

POSTGRES_DSN    = os.environ["POSTGRES_DSN"]
TRANSLATE_URL   = os.getenv("TRANSLATE_URL",    "http://83.228.244.116:8080")
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",       "5"))
CYCLE_SLEEP     = int(os.getenv("CYCLE_SLEEP",      "30"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT",  "120"))
MAX_FIELD_LEN   = int(os.getenv("MAX_FIELD_LEN",    "800"))

# Supported locales and their NLLB-200 codes
ALL_LOCALES: list[str] = ["en", "fr", "ar", "es"]

NLLB_CODE: dict[str, str] = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "ar": "arb_Arab",
    "es": "spa_Latn",
}

DEFAULT_LOCALE = "fr"   # fallback when detection fails

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("translator")

# ══════════════════════════════════════════════════════════════════════════════
# REAL-ESTATE TERM NORMALIZER
# Applied before translation so the model gets clearer input.
# ══════════════════════════════════════════════════════════════════════════════

_RE_TERMS = [
    # Tunisian S+N notation
    (re.compile(r"\bS\+0\b", re.IGNORECASE), "studio"),
    (re.compile(r"\bS\+1\b", re.IGNORECASE), "appartement 2 pièces"),
    (re.compile(r"\bS\+2\b", re.IGNORECASE), "appartement 3 pièces"),
    (re.compile(r"\bS\+3\b", re.IGNORECASE), "appartement 4 pièces"),
    (re.compile(r"\bS\+4\b", re.IGNORECASE), "appartement 5 pièces"),
    (re.compile(r"\bS\+5\b", re.IGNORECASE), "appartement 6 pièces"),
    # French F/T notation
    (re.compile(r"\b[FT]1\b", re.IGNORECASE), "studio"),
    (re.compile(r"\b[FT]2\b", re.IGNORECASE), "appartement 2 pièces"),
    (re.compile(r"\b[FT]3\b", re.IGNORECASE), "appartement 3 pièces"),
    (re.compile(r"\b[FT]4\b", re.IGNORECASE), "appartement 4 pièces"),
    (re.compile(r"\b[FT]5\b", re.IGNORECASE), "appartement 5 pièces"),
    (re.compile(r"\b[FT]6\b", re.IGNORECASE), "appartement 6 pièces"),
    # Surface
    (re.compile(r"\bm²\b"),                    "mètres carrés"),
    (re.compile(r"\bm2\b"),                    "mètres carrés"),
    # Common French abbreviations
    (re.compile(r"\bRDC\b",   re.IGNORECASE),  "rez-de-chaussée"),
    (re.compile(r"\bséj\.?\b",re.IGNORECASE),  "séjour"),
    (re.compile(r"\bch\.?\b", re.IGNORECASE),  "chambre"),
    (re.compile(r"\bsdb\.?\b",re.IGNORECASE),  "salle de bain"),
    (re.compile(r"\bwc\.?\b", re.IGNORECASE),  "toilettes"),
    (re.compile(r"\bprk\.?\b",re.IGNORECASE),  "parking"),
    (re.compile(r"\bgar\.?\b",re.IGNORECASE),  "garage"),
]


def normalize_terms(text: str) -> str:
    for pattern, replacement in _RE_TERMS:
        text = pattern.sub(replacement, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_locale(title: str | None, description: str | None) -> str:
    """
    Detect language from title + first 300 chars of description.
    Returns a 2-letter locale code ('en', 'fr', 'ar', 'es').
    Falls back to DEFAULT_LOCALE if detection fails or language unsupported.
    """
    sample = " ".join(filter(None, [title or "", (description or "")[:300]])).strip()
    if not sample:
        return DEFAULT_LOCALE
    try:
        detected = detect(sample)
        if detected in NLLB_CODE:
            return detected
        # Map similar languages to supported ones
        if detected in ("ca", "gl", "pt", "it", "ro"):
            return "es"
        return DEFAULT_LOCALE
    except LangDetectException:
        return DEFAULT_LOCALE


# ══════════════════════════════════════════════════════════════════════════════
# NLLB TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════

def _call_translate(text: str, src_lang: str, tgt_lang: str) -> str | None:
    """Single HTTP call to the NLLB translation server."""
    if src_lang == tgt_lang:
        return text
    try:
        resp = requests.post(
            f"{TRANSLATE_URL}/translate",
            json={"text": text, "src_lang": src_lang, "tgt_lang": tgt_lang},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json().get("translation", "").strip()
        return result if result else None
    except requests.exceptions.Timeout:
        log.error("NLLB timeout (%ds) %s→%s", REQUEST_TIMEOUT, src_lang, tgt_lang)
        return None
    except Exception as e:
        log.error("NLLB error: %s", e)
        return None


def translate_to_all_locales(text: str, src_locale: str) -> dict[str, str | None]:
    """
    Translate one text to all 4 locales using English as pivot.

    Flow example:
      src=fr  →  fr→en (pivot), en→ar, en→es   (fr kept as-is)
      src=en  →  kept, en→fr, en→ar, en→es
      src=ar  →  ar→en (pivot), en→fr, en→es   (ar kept as-is)
      src=es  →  es→en (pivot), en→fr, en→ar   (es kept as-is)

    Returns {locale: translated_text_or_None, ...}
    """
    if not text or not text.strip():
        return {loc: None for loc in ALL_LOCALES}

    text = normalize_terms(text.strip())[:MAX_FIELD_LEN]
    results: dict[str, str | None] = {}

    # ── Step 1: produce English pivot ────────────────────────────────────────
    if src_locale == "en":
        english = text
    else:
        english = _call_translate(text, NLLB_CODE[src_locale], NLLB_CODE["en"])
        if not english:
            # Pivot failed — cannot produce any translation
            return {loc: None for loc in ALL_LOCALES}

    results["en"] = english

    # ── Step 2: from English to all other locales ─────────────────────────────
    for locale in ["fr", "ar", "es"]:
        if locale == src_locale:
            results[locale] = text      # keep original text as-is
        else:
            results[locale] = _call_translate(
                english, NLLB_CODE["en"], NLLB_CODE[locale]
            )

    return results


def translate_features_to_all(features: list[str], src_locale: str) -> dict[str, list[str] | None]:
    """
    Translate a list of feature strings to all locales.
    Joins into one string for efficiency (single call per target locale).
    Returns {locale: [feat1, feat2, ...] or None}.
    """
    if not features:
        return {loc: None for loc in ALL_LOCALES}

    joined = ", ".join(features)
    translated = translate_to_all_locales(joined, src_locale)

    result: dict[str, list[str] | None] = {}
    for locale, text in translated.items():
        if text:
            result[locale] = [f.strip() for f in text.split(",") if f.strip()]
        else:
            result[locale] = None
    return result


# ══════════════════════════════════════════════════════════════════════════════
# POSTGRESQL QUERIES
# ══════════════════════════════════════════════════════════════════════════════

# Fetch announcements that have 0 translation rows yet
FETCH_SQL = """
    SELECT a.id,
           a.source,
           a.title,
           a.description,
           a.interior_features,
           a.exterior_features,
           a.other_features
    FROM   announcements a
    WHERE  NOT EXISTS (
               SELECT 1 FROM announcement_translations t
               WHERE  t.announcement_id = a.id
           )
    ORDER  BY a.id
    LIMIT  %s
"""

PENDING_SQL = """
    SELECT COUNT(*) AS n
    FROM   announcements a
    WHERE  NOT EXISTS (
               SELECT 1 FROM announcement_translations t
               WHERE  t.announcement_id = a.id
           )
"""

INSERT_SQL = """
    INSERT INTO announcement_translations
        (announcement_id, locale, title, description, features_translated)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (announcement_id, locale) DO UPDATE SET
        title               = EXCLUDED.title,
        description         = EXCLUDED.description,
        features_translated = EXCLUDED.features_translated,
        translated_at       = NOW()
"""


def _extract_features(row: dict) -> list[str]:
    """Merge interior/exterior/other features into a flat list of strings."""
    items: list[str] = []
    for key in ("interior_features", "exterior_features", "other_features"):
        val = row.get(key)
        if isinstance(val, list):
            items.extend(str(v) for v in val if v)
        elif isinstance(val, str) and val.strip():
            items.extend(f.strip() for f in val.split(",") if f.strip())
    return items


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CYCLE
# ══════════════════════════════════════════════════════════════════════════════

def run_cycle(pg_conn, limit: int | None = None) -> int:
    cur = pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(PENDING_SQL)
    pending = cur.fetchone()["n"]
    log.info("Pending announcements (not yet translated): %d", pending)

    if pending == 0:
        cur.close()
        return 0

    fetch_limit = min(BATCH_SIZE, limit) if limit else BATCH_SIZE
    cur.execute(FETCH_SQL, (fetch_limit,))
    rows = cur.fetchall()
    cur.close()

    processed = 0
    ins_cur = pg_conn.cursor()

    for row in rows:
        ann_id = row["id"]
        title  = row.get("title")       or ""
        desc   = row.get("description") or ""

        # ── 1. Detect source language ─────────────────────────────────────
        src_locale = detect_locale(title, desc)
        log.info(
            "  [id=%d source=%s lang=%s] %s",
            ann_id, row.get("source"), src_locale, title[:60],
        )

        # ── 2. Translate title ────────────────────────────────────────────
        title_by_locale = (
            translate_to_all_locales(title, src_locale)
            if title else {loc: None for loc in ALL_LOCALES}
        )

        # ── 3. Translate description ──────────────────────────────────────
        desc_by_locale = (
            translate_to_all_locales(desc, src_locale)
            if desc else {loc: None for loc in ALL_LOCALES}
        )

        # ── 4. Translate features ─────────────────────────────────────────
        features = _extract_features(row)
        feats_by_locale = (
            translate_features_to_all(features, src_locale)
            if features else {loc: None for loc in ALL_LOCALES}
        )

        # ── 5. Persist one row per locale ─────────────────────────────────
        for locale in ALL_LOCALES:
            feats = feats_by_locale.get(locale)
            ins_cur.execute(INSERT_SQL, (
                ann_id,
                locale,
                title_by_locale.get(locale),
                desc_by_locale.get(locale),
                json.dumps(feats, ensure_ascii=False) if feats else None,
            ))

        pg_conn.commit()
        processed += 1
        log.info("  [id=%d] ✓ %d locales saved (%d/%d)", ann_id, len(ALL_LOCALES), processed, len(rows))

        if limit and processed >= limit:
            break

    ins_cur.close()
    log.info("Cycle done — %d announcements translated", processed)
    return processed


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Universal Translator — announcements → announcement_translations"
    )
    parser.add_argument("--once",  action="store_true", help="Single pass then exit")
    parser.add_argument("--limit", type=int, default=None, help="Max announcements per run")
    args = parser.parse_args()

    log.info("═" * 60)
    log.info("UNIVERSAL TRANSLATOR  —  all sources  →  en / fr / ar / es")
    log.info("NLLB server  : %s", TRANSLATE_URL)
    log.info("Pivot        : English")
    log.info("Batch size   : %d", BATCH_SIZE)
    log.info("═" * 60)

    # Verify NLLB server is reachable
    try:
        r = requests.get(f"{TRANSLATE_URL}/health", timeout=10)
        r.raise_for_status()
        log.info("✅ NLLB server OK — %s", r.json())
    except Exception as e:
        log.error("❌ NLLB server not reachable: %s", e)
        raise SystemExit(1)

    # Connect to PostgreSQL
    pg = psycopg2.connect(POSTGRES_DSN)
    log.info("✅ PostgreSQL OK")

    if args.once or args.limit:
        run_cycle(pg, limit=args.limit)
        pg.close()
        return

    # Continuous loop
    while True:
        try:
            count = run_cycle(pg)
            if not count:
                log.info("Nothing new — sleeping %ds...", CYCLE_SLEEP)
        except Exception as e:
            log.error("Cycle error: %s — retrying in %ds", e, CYCLE_SLEEP)
            try:
                pg.rollback()
            except Exception:
                pass
            # Reconnect if connection dropped
            try:
                pg.close()
            except Exception:
                pass
            try:
                pg = psycopg2.connect(POSTGRES_DSN)
                log.info("Reconnected to PostgreSQL")
            except Exception as ce:
                log.error("Reconnect failed: %s", ce)

        time.sleep(CYCLE_SLEEP)


if __name__ == "__main__":
    main()