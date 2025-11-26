import os
from serpapi import GoogleSearch
import dspy
import re
from brand_extraction.entity_extractor import extract_entities

# --------------------------
# DSPy Signature
# --------------------------
class ExtractBrandSLM(dspy.Signature):
    prompt = dspy.InputField()
    brands = dspy.OutputField(desc="Comma-separated list of brand/company names only. No sentences.")


# --------------------------
# Product → Brand Mapping
# --------------------------
PRODUCT_BRAND_MAP = {
    "fold": "Samsung",
    "galaxy": "Samsung",
    "bravia": "Sony",
    "xperia": "Sony",
    "iphone": "Apple",
    "ipad": "Apple",
    "macbook": "Apple",
    "tiktok": "ByteDance",
    "youtube": "Google"
}
PRODUCT_REGEX = re.compile(r"\b(" + "|".join(PRODUCT_BRAND_MAP.keys()) + r")\w*\b", re.I)


class BrandAgent:

    def __init__(self, serpapi_key=None):
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.slm = dspy.Predict(ExtractBrandSLM)

    # ------------------------------------
    # STEP 1: NER extraction
    # ------------------------------------
    def ner_extract(self, prompt: str):
        ents = extract_entities(prompt)
        out = []
        for e in ents.get("entities", []):
            if e["label"] in ("ORG", "PRODUCT"):
                out.append(e["text"])
        return out

    # ------------------------------------
    # STEP 2: DSPy extraction
    # ------------------------------------
    def slm_extract(self, prompt: str):
        res = self.slm(prompt=prompt)
        return [x.strip() for x in res.brands.split(",") if x.strip()]

    # ------------------------------------
    # STEP 3: Product → Brand
    # ------------------------------------
    def resolve_products(self, prompt: str):
        matches = PRODUCT_REGEX.findall(prompt.lower())
        return [PRODUCT_BRAND_MAP[m.lower()] for m in matches]

    # ------------------------------------
    # STEP 4: SerpAPI validation
    # ------------------------------------
    def serpapi_validate(self, token: str) -> bool:
        if not self.serpapi_key:
            return True   # assume valid if SerpAPI disabled

        queries = [
            f"{token} official website",
            f"{token} brand",
            f"{token} company"
        ]

        for q in queries:
            try:
                search = GoogleSearch({
                    "q": q,
                    "engine": "google",
                    "api_key": self.serpapi_key,
                    "num": 3
                }).get_dict()

                if search.get("knowledge_graph", {}).get("type") in (
                    "Organization", "Brand", "Company"
                ):
                    return True

                if "organic_results" in search:
                    title = search["organic_results"][0].get("title", "").lower()
                    if token.lower() in title:
                        return True

            except:
                pass

        return False

    # ------------------------------------
    # CONFIDENCE CALCULATION
    # ------------------------------------
    def compute_confidence(self, brand, ner_list, slm_list, serp_valid):
        ner_score = 1.0 if brand in ner_list else 0.0
        slm_score = 1.0 if brand in slm_list else 0.0
        serp_score = 1.0 if serp_valid else 0.0

        final_score = (
            0.4 * ner_score +
            0.4 * slm_score +
            0.2 * serp_score
        )

        return round(final_score, 3)

    # ------------------------------------
    # FINAL PIPELINE
    # ------------------------------------
    def extract_brands(self, prompt: str):
        # Collect candidates
        ner_cands = self.ner_extract(prompt)
        slm_cands = self.slm_extract(prompt)
        product_cands = self.resolve_products(prompt)

        # Merge all
        raw = set(ner_cands + slm_cands + product_cands)

        results = []

        for brand in raw:
            serp_valid = self.serpapi_validate(brand)
            conf = self.compute_confidence(
                brand=brand,
                ner_list=ner_cands,
                slm_list=slm_cands,
                serp_valid=serp_valid
            )
            results.append({"brand": brand, "confidence": conf})

        # Sort brands by confidence descending
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        return results
# ============================================================