# agent/brand_agent.py
import os
import time
import re
from typing import List, Dict, Optional
from serpapi import GoogleSearch
import dspy

# -----------------------
# DSPy Chain-of-Thought Signature
# -----------------------
class BrandCoTSignature(dspy.Signature):
    """
    Chain-of-thought signature for brand extraction with reasoning trace.
    - prompt: user text
    - reasoning: optional internal reasoning (string)
    - brands: comma-separated string of extracted brands/products
    """
    prompt = dspy.InputField(desc="User prompt")
    reasoning = dspy.OutputField(desc="Model chain-of-thought reasoning")
    brands = dspy.OutputField(desc="Comma-separated brand/product candidates")


# Configure local SLM (Ollama Phi-3). Replace api_base if your Ollama runs elsewhere.
dspy.configure(lm=dspy.LM("ollama/phi3", api_base="http://localhost:11434"))


# -----------------------
# BrandAgent Implementation
# -----------------------
class BrandAgent:
    """
    Advanced BrandAgent:
     - Uses Chain-of-Thought (dspy.ChainOfThought) to extract product/brand tokens.
     - Performs SerpAPI product search, KG parsing, and image-title parsing to resolve brand.
     - Uses query expansion and multi-source evidence for confidence scoring.
    """
    def __init__(self, serpapi_key: Optional[str] = None, pause_between_calls: float = 0.4):
        self.predictor = dspy.ChainOfThought(BrandCoTSignature)
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.pause = pause_between_calls

        # small list of product-expansion suffixes to try
        self.query_suffixes = ["", "phone", "smartphone", "device", "model", "product", "mobile", "gadget"]
        # small list of known brand token patterns to normalize (can extend)
        self.brand_normalize_patterns = [
            (re.compile(r"\bgalaxy\s*z?\s*fold\b", re.I), "Samsung"),
            (re.compile(r"\biphone\b", re.I), "Apple"),
            (re.compile(r"\bpixel\b", re.I), "Google"),
            (re.compile(r"\bps5\b", re.I), "Sony"),
            # add more patterns as needed
        ]

    # -----------------------
    # Public API
    # -----------------------
    def extract_brands(self, prompt: str) -> Dict:
        """
        Main entrypoint.
        Returns dict:
        {
          "brands": [<brand names>],
          "confidence": 0-1 float,
          "method": str,
          "evidence": [ {source, snippet, link, score}, ... ],
          "reasoning": <string from ChainOfThought>
        }
        """
        # 1) Use Chain-of-Thought to get candidate tokens (product-like or brand-like)
        cot_res = self._run_cot(prompt)
        reasoning = getattr(cot_res, "reasoning", "") or ""
        raw_candidates = getattr(cot_res, "brands", "") or ""
        candidate_tokens = self._normalize_candidates(raw_candidates)

        # If COT produced nothing, also try a direct lightweight tokenization fallback
        if not candidate_tokens:
            candidate_tokens = self._heuristic_token_candidates(prompt)

        # If no serpapi key -> LLM-only fallback
        if not self.serpapi_key:
            # low confidence since no external verification
            return {
                "brands": candidate_tokens,
                "confidence": round(0.45 if candidate_tokens else 0.0, 2),
                "method": "SLM ChainOfThought (LLM-only, no SerpAPI)",
                "evidence": [],
                "reasoning": reasoning
            }

        # 2) Expand queries and gather evidence from multiple SerpAPI sources
        evidence = []
        resolved_brands = set()

        # Try product search first (highest-value signal)
        for token in candidate_tokens:
            prod_res = self._serp_product_search(token)
            time.sleep(self.pause)
            if prod_res:
                # product_results might be a dict or list, handle both
                brand = self._parse_product_brand(prod_res)
                if brand:
                    resolved_brands.add(brand)
                    evidence.append({
                        "source": "google_product",
                        "token": token,
                        "brand": brand,
                        "detail": prod_res,
                        "score": 0.9
                    })

        # 3) For tokens not resolved by product search, try knowledge graph + organic results + images
        for token in candidate_tokens:
            # Knowledge Graph / regular google
            kg_res = self._serp_kg_search(token)
            time.sleep(self.pause)
            kg_brand = self._parse_kg_brand(kg_res)
            if kg_brand:
                resolved_brands.add(kg_brand)
                evidence.append({
                    "source": "knowledge_graph",
                    "token": token,
                    "brand": kg_brand,
                    "detail": kg_res,
                    "score": 0.8
                })
                continue  # KG likely sufficient

            # Organic titles/snippets
            organic_brand = self._parse_organic_for_brand(kg_res)
            if organic_brand:
                resolved_brands.add(organic_brand)
                evidence.append({
                    "source": "organic_results",
                    "token": token,
                    "brand": organic_brand,
                    "detail": kg_res.get("organic_results", []) if isinstance(kg_res, dict) else {},
                    "score": 0.6
                })

            # Image titles/captions
            img_res = self._serp_image_search(token)
            time.sleep(self.pause)
            img_brand = self._parse_image_titles_for_brand(img_res)
            if img_brand:
                resolved_brands.add(img_brand)
                evidence.append({
                    "source": "image_titles",
                    "token": token,
                    "brand": img_brand,
                    "detail": img_res,
                    "score": 0.6
                })

        # 4) Normalize resolved brands by pattern mapping and simple title-case
        normalized = self._normalize_brand_set(resolved_brands)

        # 5) Confidence scoring: weighted average of evidence scores
        confidence = self._compute_confidence(evidence, fallback_candidates=candidate_tokens)

        # 6) If nothing resolved, still attempt pattern-based normalization from tokens
        if not normalized and candidate_tokens:
            for token in candidate_tokens:
                p = self._pattern_map_brand(token)
                if p:
                    normalized.add(p)
                    evidence.append({"source": "pattern_map", "token": token, "brand": p, "score": 0.5})
            confidence = self._compute_confidence(evidence, fallback_candidates=candidate_tokens)

        # 7) Final packaging
        method_parts = []
        if candidate_tokens:
            method_parts.append("SLM ChainOfThought")
        if any(e["source"] == "google_product" for e in evidence):
            method_parts.append("SerpAPI product_search")
        if any(e["source"] == "knowledge_graph" for e in evidence):
            method_parts.append("KG_verification")
        if any(e["source"] == "image_titles" for e in evidence):
            method_parts.append("image_title_parsing")
        if not self.serpapi_key:
            method_parts = ["SLM ChainOfThought (LLM-only, no SerpAPI)"]

        method = " + ".join(method_parts) if method_parts else "SLM ChainOfThought"

        return {
            "brands": sorted(list(normalized)),
            "confidence": round(confidence, 2),
            "method": method,
            "evidence": evidence,
            "reasoning": reasoning
        }

    # -----------------------
    # Helper: run Chain-of-Thought to get candidate tokens
    # -----------------------
    def _run_cot(self, prompt: str):
        """
        Returns Chain-of-Thought result (with .reasoning and .brands attributes).
        The ChainOfThought signature expects the model to return a short reasoning and
        possibly a comma-separated list of brand/product tokens.
        """
        try:
            # Provide a short instruction in the prompt to improve extraction reliability
            cot_prompt = (
                f"{prompt}\n\n"
                "Task: List any product model names or brand identifiers mentioned or implied in the prompt. "
                "If no explicit brand is present, extract the product model or device name only (comma-separated). "
                "Also include a short 'REASONING:' explanation before the final list.\n\n"
                "Output format:\nREASONING: <text>\nBRANDS: <comma separated list>\n"
            )
            res = self.predictor(prompt=cot_prompt)
            return res
        except Exception as e:
            # graceful fallback: return an object-like with empty fields
            class _R: pass
            r = _R()
            r.reasoning = f"ChainOfThought failed: {e}"
            r.brands = ""
            return r

    # -----------------------
    # Helper: normalize raw brand string from model to token list
    # -----------------------
    def _normalize_candidates(self, raw: str) -> List[str]:
        if not raw:
            return []
        # try to extract after "BRANDS:" if the model followed the format
        m = re.search(r"BRANDS\s*[:\-]\s*(.+)$", raw, flags=re.I | re.S)
        if m:
            raw = m.group(1)
        # split on common separators
        tokens = re.split(r"[,\n;/\|]+", raw)
        tokens = [t.strip() for t in tokens if t.strip()]
        # remove obvious non-product words
        tokens = [t for t in tokens if len(t) <= 60]
        return tokens

    # -----------------------
    # Fallback heuristic candidate extraction
    # -----------------------
    def _heuristic_token_candidates(self, prompt: str) -> List[str]:
        # Look for token patterns: model names (alphanumeric + digits), capitalized words near digits
        tokens = set()
        # words with digits like "Fold7", "ZFold 7", "S21"
        for match in re.finditer(r"\b([A-Za-z]{2,}\s*[A-Za-z0-9-]*\d{1,4}[A-Za-z0-9-]*)\b", prompt):
            tokens.add(match.group(1).strip())
        # also capture capitalized words (possible brands)
        for match in re.finditer(r"\b([A-Z][a-zA-Z0-9]{2,})\b", prompt):
            tokens.add(match.group(1).strip())
        return list(tokens)

    # -----------------------
    # SERPAPI: product search
    # -----------------------
    def _serp_product_search(self, query: str) -> Optional[Dict]:
        try:
            params = {
                "engine": "google_product",
                "q": query,
                "api_key": self.serpapi_key,
                "num": 3
            }
            s = GoogleSearch(params)
            res = s.get_dict()
            return res
        except Exception:
            return None

    # -----------------------
    # SERPAPI: knowledge graph / regular google search
    # -----------------------
    def _serp_kg_search(self, query: str) -> Dict:
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": 5
            }
            s = GoogleSearch(params)
            return s.get_dict()
        except Exception:
            return {}

    # -----------------------
    # SERPAPI: image search
    # -----------------------
    def _serp_image_search(self, query: str) -> Dict:
        try:
            params = {
                "engine": "google_images",
                "q": query,
                "api_key": self.serpapi_key,
                "num": 5
            }
            s = GoogleSearch(params)
            return s.get_dict()
        except Exception:
            return {}

    # -----------------------
    # Parsers for responses
    # -----------------------
    def _parse_product_brand(self, prod_res: Dict) -> Optional[str]:
        """
        Parse product_results to find brand/manufacturer.
        Handles a variety of SerpAPI product_result shapes.
        """
        if not prod_res:
            return None
        # Some responses have "product_results" key as dict
        pr = prod_res.get("product_results") or prod_res.get("product_result")
        if isinstance(pr, dict):
            # common keys: title, brand, extensions
            brand = pr.get("brand") or pr.get("manufacturer")
            if brand:
                return self._clean_brand(brand)
            # sometimes title contains brand
            title = pr.get("title", "")
            b = self._extract_brand_from_title(title)
            if b:
                return b
        if isinstance(pr, list):
            for item in pr:
                brand = item.get("brand") or item.get("manufacturer")
                if brand:
                    return self._clean_brand(brand)
                title = item.get("title", "")
                b = self._extract_brand_from_title(title)
                if b:
                    return b
        # fallback: sometimes response has "extensions" or knowledge graph
        kg = prod_res.get("knowledge_graph") or {}
        if kg:
            brand = kg.get("brand") or kg.get("manufacturer") or kg.get("title")
            if brand:
                return self._clean_brand(brand)
        return None

    def _parse_kg_brand(self, kg_res: Dict) -> Optional[str]:
        if not kg_res:
            return None
        kg = kg_res.get("knowledge_graph")
        if isinstance(kg, dict):
            brand = kg.get("brand") or kg.get("manufacturer")
            if brand:
                return self._clean_brand(brand)
            # sometimes title signals brand/product family
            title = kg.get("title") or ""
            b = self._extract_brand_from_title(title)
            if b:
                return b
        # also check top organic results titles/snippets
        for r in kg_res.get("organic_results", [])[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            b = self._extract_brand_from_text(title) or self._extract_brand_from_text(snippet)
            if b:
                return b
        return None

    def _parse_organic_for_brand(self, res: Dict) -> Optional[str]:
        if not res:
            return None
        for r in res.get("organic_results", [])[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            candidate = self._extract_brand_from_text(title) or self._extract_brand_from_text(snippet)
            if candidate:
                return candidate
        return None

    def _parse_image_titles_for_brand(self, img_res: Dict) -> Optional[str]:
        if not img_res:
            return None
        # image results often have 'title' or 'snippet' fields
        for item in img_res.get("image_results", [])[:6]:
            title = item.get("title") or item.get("snippet") or ""
            candidate = self._extract_brand_from_text(title)
            if candidate:
                return candidate
        # fallback: inspect 'source' or 'description'
        for item in img_res.get("inline_images", [])[:6]:
            title = item.get("title") or item.get("alt") or ""
            candidate = self._extract_brand_from_text(title)
            if candidate:
                return candidate
        return None

    # -----------------------
    # Small NLP helpers for brand extraction from text
    # -----------------------
    def _extract_brand_from_title(self, title: str) -> Optional[str]:
        # If title has patterns like "Samsung Galaxy Z Fold 7", "Apple iPhone 15"
        # we try to extract the first proper-brand-like token
        if not title:
            return None
        # common pattern: "<Brand> <Product...>"
        m = re.match(r"^\s*([A-Z][A-Za-z0-9&\.\-]{1,30})\b", title)
        if m:
            return self._clean_brand(m.group(1))
        return None

    def _extract_brand_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        # simple heuristics: check for known brand tokens in text
        # try direct match against normalized pattern list
        for pat, brand in self.brand_normalize_patterns:
            if pat.search(text):
                return brand
        # otherwise try to find capitalized brand-like token
        m = re.search(r"\b([A-Z][A-Za-z0-9&\.\-]{2,30})\b", text)
        if m:
            return self._clean_brand(m.group(1))
        return None

    def _clean_brand(self, b: str) -> str:
        return re.sub(r"[^A-Za-z0-9&\.\- ]+", "", b).strip().title()

    def _pattern_map_brand(self, token: str) -> Optional[str]:
        if not token:
            return None
        for pat, brand in self.brand_normalize_patterns:
            if pat.search(token):
                return brand
        return None

    def _normalize_brand_set(self, brands_set: set) -> set:
        out = set()
        for b in brands_set:
            if not b:
                continue
            out.add(self._clean_brand(b))
        return out

    # -----------------------
    # Confidence computation
    # -----------------------
    def _compute_confidence(self, evidence: List[Dict], fallback_candidates: List[str]) -> float:
        """
        Weighted scoring:
         - product_search evidence: weight 0.9
         - KG evidence: weight 0.8
         - organic/image: weight 0.6
         - pattern_map: 0.5
         - if no evidence but there are fallback candidates -> low confidence 0.35
        Final confidence scaled to [0, 1].
        """
        if not evidence:
            return 0.35 if fallback_candidates else 0.0
        # compute normalized weighted average of evidence scores
        total_w = 0.0
        total_score = 0.0
        for e in evidence:
            score = e.get("score", 0.5)
            total_score += score
            total_w += 1.0
        avg = (total_score / total_w) if total_w else 0.0
        # clamp
        return max(0.0, min(1.0, float(avg)))

