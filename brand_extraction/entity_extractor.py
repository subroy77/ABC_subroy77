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
