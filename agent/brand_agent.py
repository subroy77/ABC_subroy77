import os
import dspy
from serpapi import GoogleSearch

# Define the task structure
class BrandExtractionSignature(dspy.Signature):
    """Extract brand names from a user prompt"""
    prompt = dspy.InputField(desc="The user query or prompt text")
    brands = dspy.OutputField(desc="List of brand names mentioned in the text")

# Configure LLM once
dspy.configure(lm=dspy.LM("ollama/phi3"))

class BrandAgent:
    def __init__(self):
        # Create the predictor using the signature
        self.predictor = dspy.Predict(BrandExtractionSignature)

    def extract_brands(self, prompt: str):
        result = self.predictor(prompt=prompt)
        brand_list = [b.strip() for b in result.brands.split(",") if b.strip()]

        validated = []
        for brand in brand_list:
            params = {
                "engine": "google",
                "q": f"{brand} official site",
                "api_key": os.getenv("SERPAPI_API_KEY")
            }
            search = GoogleSearch(params)
            res = search.get_dict()
            if "organic_results" in res:
                validated.append(brand)

        return validated