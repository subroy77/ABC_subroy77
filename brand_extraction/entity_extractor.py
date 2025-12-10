import json
import re

def safe_extract_brands(llm_response: str):
    """
    Parse brand + confidence JSON returned by LLM.
    Clean the response if it contains reasoning text, markdown fences, or malformed JSON.

    Expected return structure:
    [
        {"brand": "Samsung", "confidence": 0.97},
        {"brand": "LG", "confidence": 0.94}
    ]
    """

    if not llm_response or not isinstance(llm_response, str):
        return []

    # Strip code fences like ```json ... ```
    cleaned = re.sub(r"```(?:json)?", "", llm_response, flags=re.IGNORECASE).strip("` \n")

    # Remove <reasoning> ... </reasoning> or similar LLM thinking content
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL).strip()

    try:
        # Some models return dict with key "brands"
        parsed = json.loads(cleaned)
    except Exception:
        # Try extracting potential JSON object manually
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                return []
        else:
            return []

    # Standardize expected list format
    if isinstance(parsed, dict) and "brands" in parsed and isinstance(parsed["brands"], list):
        brands = parsed["brands"]
    elif isinstance(parsed, list):
        brands = parsed
    else:
        return []

    normalized = []
    for b in brands:
        if not isinstance(b, dict) or "brand" not in b:
            continue
        brand_name = str(b["brand"]).strip()
        conf = float(b.get("confidence", 0.0))
        normalized.append({
            "brand": brand_name,
            "confidence": max(0.0, min(conf, 1.0))  # clamp 0→1
        })

    return normalized

'''
import spacy
from spacy.cli import download

# Upgrade to transformer-based model for better NER
MODEL_NAME = "en_core_web_trf"
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME)

def extract_entities(sentence):
    doc = nlp(sentence)
    result = {
        "tokens": [],
        "entities": [],
        "pos": {},
    }
    # POS tagging
    for token in doc:
        pos_tag = token.pos_
        if pos_tag not in result["pos"]:
            result["pos"][pos_tag] = []
        result["pos"][pos_tag].append(token.text)
        result["tokens"].append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop
        })
    # Named Entity Recognition
    for ent in doc.ents:
        result["entities"].append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return result

if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    output = extract_entities(sentence)
    import pprint
    pprint.pprint(output)
'''

