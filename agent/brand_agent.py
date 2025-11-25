# import os
# import re
# from typing import List, Dict, Optional
# import numpy as np
# from serpapi import GoogleSearch
# import dspy
# from sentence_transformers import SentenceTransformer

# from brand_extraction.entity_extractor import extract_entities

# # ------------------------------------------------------------
# # Load embedding model for robust brand matching
# # ------------------------------------------------------------
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# KNOWN_BRANDS = [
#     "Samsung","Apple","LG","Sony","Philips","Xiaomi","OnePlus","Google",
#     "Nokia","Motorola","Dell","HP","Lenovo","Asus","Acer","Microsoft",
#     "Realme","Oppo","Vivo"
# ]
# KNOWN_BRAND_EMB = embed_model.encode(KNOWN_BRANDS)

# def cosine_sim(a,b):
#     return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

# def vector_validate(token: str) -> Optional[str]:
#     emb = embed_model.encode([token])[0]
#     sims = [cosine_sim(emb, kb) for kb in KNOWN_BRAND_EMB]
#     best = int(np.argmax(sims))
#     if sims[best] > 0.55:
#         return KNOWN_BRANDS[best]
#     return None


# # ------------------------------------------------------------
# # Product → Brand Mapping
# # ------------------------------------------------------------
# PRODUCT_BRAND_MAP = {
#     "fold": "Samsung", "galaxy": "Samsung", "note": "Samsung",
#     "iphone": "Apple", "ipad": "Apple", "macbook": "Apple",
#     "pixel": "Google", "redmi": "Xiaomi", "mi": "Xiaomi",
#     "poco": "Xiaomi", "oneplus": "OnePlus",
#     "moto": "Motorola", "motorola": "Motorola"
# }

# PRODUCT_REGEX = re.compile(
#     r"\b(fold|galaxy|note|pixel|iphone|ipad|macbook|series|moto|oneplus|nokia|mi|redmi|poco)\s*\d*[a-zA-Z0-9-]*\b",
#     re.I
# )


# # ------------------------------------------------------------
# # DSPy Signature (restricted)
# # ------------------------------------------------------------
# class BrandDisambiguation(dspy.Signature):
#     prompt = dspy.InputField()
#     candidates = dspy.InputField()
#     # only return VALID BRANDS as comma-separated tokens
#     final_brands = dspy.OutputField(desc="Return ONLY brand names, comma-separated. No sentences.")


# class BrandAgent:
#     def __init__(self, serpapi_key=None):
#         self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
#         self.cot = dspy.Predict(BrandDisambiguation)

#     # --------------------------------------------------------
#     # Step 1 — Extract candidates using NER + Regex
#     # --------------------------------------------------------
#     def extract_candidates(self, prompt: str) -> List[str]:
#         ents = extract_entities(prompt)
#         cands = set()

#         for ent in ents.get("entities", []):
#             if ent["label"] in ("ORG", "PRODUCT"):
#                 cands.add(ent["text"])

#         # product model tokens like “fold 7”
#         for m in PRODUCT_REGEX.findall(prompt):
#             cands.add(m)

#         return list(cands)

#     # --------------------------------------------------------
#     # Step 2 — Normalize tokens into real brands
#     # --------------------------------------------------------
#     def normalize_candidates(self, tokens: List[str], prompt: str) -> List[str]:
#         final = set()

#         for t in tokens:
#             low = t.lower()

#             # product → brand mapping
#             for key, brand in PRODUCT_BRAND_MAP.items():
#                 if key in low:
#                     final.add(brand)

#             # vector similarity fallback
#             guess = vector_validate(t)
#             if guess:
#                 final.add(guess)

#         # also infer directly from text
#         for b in KNOWN_BRANDS:
#             if b.lower() in prompt.lower():
#                 final.add(b)

#         return list(final)

#     # --------------------------------------------------------
#     # Step 3 — Light evidence scoring
#     # --------------------------------------------------------
#     def gather_evidence(self, brands: List[str]) -> Dict[str, int]:
#         scores = {b: 0 for b in brands}

#         if not self.serpapi_key:
#             return scores

#         for b in brands:
#             try:
#                 search = GoogleSearch({
#                     "engine": "google",
#                     "q": f"{b} official",
#                     "num": 2,
#                     "api_key": self.serpapi_key
#                 }).get_dict()

#                 if "organic_results" in search:
#                     scores[b] += len(search["organic_results"])
#             except:
#                 pass

#         return scores

#     # --------------------------------------------------------
#     # Step 4 — Final DSPy CoT (restricted)
#     # --------------------------------------------------------
#     def cot_filter(self, prompt: str, brands: List[str]) -> List[str]:
#         if not brands:
#             return []

#         cand_str = ",".join(brands)

#         try:
#             out = self.cot(prompt=prompt, candidates=cand_str)
#             raw = out.final_brands

#             # FORCE CLEAN LIST
#             clean = [b for b in KNOWN_BRANDS if b.lower() in raw.lower()]
#             if clean:
#                 return clean
#         except:
#             pass

#         return brands  # fallback

#     # --------------------------------------------------------
#     # Full Pipeline
#     # --------------------------------------------------------
#     def extract_brands(self, prompt: str) -> List[str]:
#         candidates = self.extract_candidates(prompt)
#         normalized = self.normalize_candidates(candidates, prompt)

#         # evidence scoring
#         scores = self.gather_evidence(normalized)

#         # rank brands
#         ranked = sorted(normalized, key=lambda b: -scores[b])

#         # DSPy final filter
#         final = self.cot_filter(prompt, ranked)

#         return list(dict.fromkeys(final))

# import dspy

# # Define signature
# class BrandExtractionSignature(dspy.Signature):
#     """Extract the Brand name from user text"""
#     prompt = dspy.InputField(desc="The text containing pbrand name")
#     brand = dspy.OutputField(desc="Brand name like Samsung, Apple, etc.")

# class BrandAgent:
#     def __init__(self):
#         self.predictor = dspy.Predict(BrandExtractionSignature)

#     def extract_brand(self, prompt: str):
#         result = self.predictor(prompt=prompt)
#         return result.brand.strip()

import dspy
from brand_extraction.entity_extractor import extract_entities

# dspy signature for validating + cleaning brand candidates
class BrandValidationSignature(dspy.Signature):
    """Given entity candidates, return only real brands."""
    entities = dspy.InputField(desc="List of entity candidates extracted using NER")
    text = dspy.InputField(desc="Original user query")
    brands = dspy.OutputField(desc="A clean list of brand names present in the text")


class BrandAgent:
    def __init__(self):
        self.validator = dspy.Predict(BrandValidationSignature)

    def extract_brand(self, prompt: str):
        # 1. Extract raw entities (SpaCy)
        ner_result = extract_entities(prompt)
        candidates = [ent["text"] for ent in ner_result["entities"]]

        # Remove duplicates
        candidates = list(set(candidates))

        # If no NER entities found, fallback to DSPy directly
        if not candidates:
            result = self.validator(entities=[], text=prompt)
            return result.brands

        # 2. Ask DSPy to filter which of these are actual brands
        result = self.validator(
            entities=candidates,
            text=prompt
        )

        # Output is a list
        if isinstance(result.brands, str):
            # convert comma separated → list
            brands = [b.strip() for b in result.brands.split(",") if b.strip()]
        else:
            brands = result.brands

        return brands
