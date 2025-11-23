import os
import re
from typing import List, Dict, Optional

import dspy
from serpapi import GoogleSearch

# Optional: spaCy for NER
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

# -----------------------
# DSPy Signatures (predictor + chain of thought)
# -----------------------
class BrandExtractionSignature(dspy.Signature):
    """Extract brand names from a user prompt"""
    prompt = dspy.InputField(desc="The user query or prompt text")
    brands = dspy.OutputField(desc="List of brand names mentioned in the text")

class BrandDisambiguationCoT(dspy.Signature):
    """Use chain-of-thought to decide final brand from NER + KG evidence"""
    prompt = dspy.InputField(desc="Original user prompt")
    candidates = dspy.InputField(desc="Comma-separated candidate tokens from NER and heuristics")
    evidence = dspy.InputField(desc="Short list of evidence snippets (comma separated)")
    final_brand = dspy.OutputField(desc="Final chosen brand name or empty")

# Configure LLM once (adjust to your environment)
# Example: dspy.configure(lm=dspy.LM("ollama/phi3"))


class BrandAgent:
    def __init__(self, serpapi_key: Optional[str] = None, use_spacy: bool = True):
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")

        # predictor for quick extraction using dspy signature (could be used for direct prompting)
        self.predictor = dspy.Predict(BrandExtractionSignature)
        # chain-of-thought predictor for disambiguation
        self.cot = dspy.Predict(BrandDisambiguationCoT)

        # spaCy NER model (optional fallback)
        self.nlp = None
        if use_spacy and _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                # user may need to run: python -m spacy download en_core_web_sm
                self.nlp = None

        # heuristics regex for product/model patterns (e.g. 'fold 7', 'iphone 14')
        self.model_regex = re.compile(r"\b(?:fold|pixel|iphone|galaxy|note|series|moto|oneplus|nokia|mi|redmi|poco)\s*\d+[a-zA-Z0-9-]*\b", re.I)

    # -----------------------
    # 1) NER + heuristic candidate extraction
    # -----------------------
    def extract_candidates(self, prompt: str) -> List[str]:
        print("Extracting brand candidates using NER and heuristics...")
        candidates = set()

        # 1.a spaCy NER (ORG / PRODUCT / PERSON sometimes) if available
        if self.nlp:
            doc = self.nlp(prompt)
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PRODUCT", "PERSON"):
                    candidates.add(ent.text.strip())

        # 1.b heuristic regex for models ("fold 7", "iphone 14")
        for m in self.model_regex.findall(prompt):
            candidates.add(m.strip())

        # 1.c token-level heuristics: capitalized words that look like brands
        tokens = re.findall(r"\b[A-Z][a-zA-Z0-9&+-]{2,}\b", prompt)
        for t in tokens:
            # filter out common sentence starts
            if t.lower() not in ("what", "when", "where", "who", "how", "my"):
                candidates.add(t)

        # 1.d fallback: ask dspy quick predictor to extract any brand tokens directly
        try:
            quick = self.predictor(prompt=prompt)
            if hasattr(quick, 'brands') and quick.brands:
                for b in re.split(r",|;|\\n", quick.brands):
                    if b.strip():
                        candidates.add(b.strip())
        except Exception:
            # ignore: dspy predictor is optional here
            pass

        return list(candidates)

    # -----------------------
    # 2) Build a lightweight KG / evidence list from SerpApi
    #    For each candidate or model token, do targeted searches and collect
    #    1-2 top snippets that link candidate -> brand / manufacturer
    # -----------------------
    def gather_evidence(self, tokens: List[str], top_n: int = 3) -> Dict[str, List[str]]:
        print("Gathering evidence from SerpApi...")
        evidence = {t: [] for t in tokens}
        if not self.serpapi_key:
            # If no SerpApi key available, return empty evidence; downstream CoT will note that
            return evidence

        for token in tokens:
            queries = [
                f"{token} manufacturer",
                f"{token} phone manufacturer",
                f"{token} official site",
                f"{token} who makes {token}",
            ]

            snippets = []
            for q in queries:
                params = {
                    "engine": "google",
                    "q": q,
                    "api_key": self.serpapi_key,
                    "num": top_n,
                }
                try:
                    search = GoogleSearch(params)
                    res = search.get_dict()
                except Exception:
                    res = {}

                # collect short evidence: title + snippet OR domain
                if res and "organic_results" in res:
                    for r in res.get("organic_results", [])[:top_n]:
                        title = r.get("title") or ""
                        snippet = r.get("snippet") or r.get("rich_snippet", {}).get("top", "")
                        domain = r.get("displayed_link") or r.get("link") or ""
                        entry = " - ".join(p for p in (title.strip(), snippet.strip(), domain.strip()) if p)
                        if entry:
                            snippets.append(entry)

                # If we already have enough snippets for this token, break
                if len(snippets) >= top_n:
                    break

            evidence[token] = snippets

        return evidence

    # -----------------------
    # 3) Use Chain-of-Thought to decide final brands from candidates + KG evidence
    # -----------------------
    def disambiguate_with_cot(self, prompt: str, candidates: List[str], evidence: Dict[str, List[str]]) -> List[str]:
        print("Predicting brands using Chain-of-Thought...")
        # prepare inputs
        cand_str = ", ".join(candidates) if candidates else ""
        # flatten evidence to short list of strings
        ev_list = []
        for c in candidates:
            ev_list.extend([f"{c}: {s}" for s in evidence.get(c, [])[:2]])
        ev_str = " | ".join(ev_list)

        try:
            cot_res = self.cot(prompt=prompt, candidates=cand_str, evidence=ev_str)
            out = getattr(cot_res, 'final_brand', None)
            if out:
                # support comma separated output
                final = [b.strip() for b in re.split(r",|;", out) if b.strip()]
                return final
        except Exception:
            # If CoT fails, fallback to simple evidence heuristics below
            pass

        # -----------------------
        # 3.b Fallback heuristic merging: if a candidate's evidence contains known brand tokens
        # we'll pick the brand with the strongest evidence. Very light scoring.
        # -----------------------
        brand_scores: Dict[str, int] = {}
        for token in candidates:
            score = 0
            for s in evidence.get(token, []):
                # prefer official / manufacturer domains and brand mentions
                score += bool(re.search(r"official site|manufacturer|by\s+[A-Z][a-zA-Z0-9]+", s, re.I)) * 3
                score += bool(re.search(r"samsung|apple|lg|oneplus|google|xiaomi|motorola|nokia|huawei", s, re.I)) * 5
                score += 1  # small boost for any evidence

            brand_scores[token] = score

        # choose tokens with score > 0 ordered by score
        chosen = [t for t, sc in sorted(brand_scores.items(), key=lambda x: -x[1]) if sc > 0]

        # optionally, map model tokens like 'fold 7' -> manufacturer by running a targeted search
        final_brands = []
        for t in chosen:
            # if token already looks like a brand (capitalized single-word), keep it
            if re.match(r"^[A-Z][a-zA-Z0-9&+-]{2,}$", t):
                final_brands.append(t)
                continue

            # else, inspect evidence strings for brand names using regex
            evs = evidence.get(t, [])
            mapped = None
            for s in evs:
                m = re.search(r"(Samsung|Apple|LG|OnePlus|Google|Xiaomi|Motorola|Nokia|Huawei)", s, re.I)
                if m:
                    mapped = m.group(1)
                    break

            if mapped:
                final_brands.append(mapped)
            else:
                # if token itself contains a known brand substring
                for b in ["samsung", "apple", "lg", "oneplus", "google", "xiaomi", "motorola", "nokia", "huawei"]:
                    if b in t.lower():
                        final_brands.append(b.capitalize())
                        break

                # --- Improved dynamic brand resolution (no hardcoding) ---
        resolved_brands = set()
        for c in list(candidates):
            token = c.strip()
            # Query SerpApi: "<token> brand" or "who makes <token>"
            if self.serpapi_key:
                queries = [f"{token} brand", f"who makes {token}", f"{token} manufacturer"]
                for q in queries:
                    try:
                        params = {"engine": "google", "q": q, "api_key": self.serpapi_key, "num": 3}
                        res = GoogleSearch(params).get_dict()
                    except Exception:
                        continue
                    if not res or "organic_results" not in res:
                        continue
                    for r in res.get("organic_results", [])[:3]:
                        txt = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                        # Extract any capitalized brand from the snippet
                        m = re.findall(r"([A-Z][a-zA-Z0-9]+)", r.get("title", "") + " " + r.get("snippet", ""))
                        for b in m:
                            if b.lower() not in [t.lower() for t in candidates]:
                                resolved_brands.add(b)
            # Fallback: dspy-CoT reasoning to guess brand from token meaning
            try:
                cot_res = self.cot(prompt=prompt, candidates=token, evidence=f"token={token}")
                out = getattr(cot_res, 'final_brand', None)
                if out:
                    for b in re.split(r",|;", out):
                        if b.strip():
                            resolved_brands.add(b.strip())
            except Exception:
                pass

        candidates.extend(list(resolved_brands))

        # dedupe and return
        return list(dict.fromkeys(final_brands))

    # -----------------------
    # Public method: full pipeline
    # -----------------------
    def extract_brands(self, prompt: str) -> List[str]:
        print("Extracting brands from prompt...")
        # 1) get candidates from NER + heuristics
        candidates = self.extract_candidates(prompt)

        # HARD FIX: explicitly map known Apple product prefixes → Apple
        apple_products = ["ipad", "ipad pro", "iphone", "macbook", "imac", "mac mini", "mac studio", "ipad air"]
        for ap in apple_products:
            if ap in prompt.lower():
                candidates.append("Apple")

        # HARD FIX: explicitly map Samsung product families → Samsung
        samsung_products = ["galaxy", "galaxy tab", "tab s", "tab a", "note", "s22", "fold", "flip", "galaxy tab s", "tab"]
        for sp in samsung_products:
            if sp in prompt.lower():
                candidates.append("Samsung")

        # dedupe
        candidates = list(dict.fromkeys(candidates))

        # 2) build evidence via SerpApi
        evidence = self.gather_evidence(candidates)

        # 3) disambiguate using CoT + heuristics
        final = self.disambiguate_with_cot(prompt, candidates, evidence)

        # ensure explicit mapping output
        if "Apple" not in final:
            for ap in apple_products:
                if ap in prompt.lower():
                    final.append("Apple")
                    break
        if "Samsung" not in final:
            for sp in samsung_products:
                if sp in prompt.lower():
                    final.append("Samsung")
                    break

        return list(dict.fromkeys(final))


# -----------------------
# Vector Similarity Brand Validation (Added)
# -----------------------
import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

KNOWN_BRANDS = ["Apple", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo", "Asus", "Acer", "Microsoft", "Google", "Xiaomi", "OnePlus", "Realme"]
KNOWN_BRAND_EMB = embed_model.encode(KNOWN_BRANDS)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def validate_brand_with_vector_similarity(candidate: str):
    print(f"Validating brand candidate '{candidate}' with vector similarity...")
    cand_emb = embed_model.encode([candidate])[0]
    sims = [cosine_sim(cand_emb, kb) for kb in KNOWN_BRAND_EMB]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] > 0.55:
        return KNOWN_BRANDS[best_idx]
    return None

