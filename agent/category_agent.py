import dspy

# Define signature
class CategoryExtractionSignature(dspy.Signature):
    """Extract the product or service category from user text"""
    prompt = dspy.InputField(desc="The text containing product or service info")
    category = dspy.OutputField(desc="Product category like electronics, fashion, etc.")

class CategoryAgent:
    def __init__(self):
        self.predictor = dspy.Predict(CategoryExtractionSignature)

    def extract_category(self, prompt: str):
        result = self.predictor(prompt=prompt)
        return result.category.strip()