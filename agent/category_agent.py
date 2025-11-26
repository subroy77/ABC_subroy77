import dspy
import re

# ---------------------------------------------------------
# Controlled taxonomy
# ---------------------------------------------------------
CONTROLLED_CATEGORIES = {
    "Electronics": [
        "Smartphone", "Phone", "Mobile",
        "TV", "Television", "LED", "OLED",
        "Laptop", "Computer", "PC",
        "Tablet", "iPad",
        "Headphones", "Earbuds", "Speakers",
        "Refrigerator", "Fridge",
        "Washing Machine",
        "Camera", "DSLR",
        "Smartwatch", "Wearable"
    ],

    "Household": [
        "Bedsheet", "Mattress", "Pillow",
        "Kitchenware", "Cookware", "Utensils",
        "Furniture", "Sofa", "Table"
    ],

    "Personal Care": [
        "Perfume", "Deodorant",
        "Soap", "Shampoo",
        "Skincare", "Makeup"
    ],

    "Entertainment": [
        "Streaming", "OTT", "Movies",
        "Music", "Game", "Gaming", "Console"
    ],

    "Education": [
        "Training", "Course",
        "Learning", "E-learning",
        "Books", "Study", "Podcast"
    ]
}

# ---------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------
class CategoryExtractionSignature(dspy.Signature):
    prompt = dspy.InputField(desc="User query text")
    category = dspy.OutputField(desc="Predicted category")


# ---------------------------------------------------------
# Category Agent
# ---------------------------------------------------------
class CategoryAgent:
    def __init__(self):
        self.predictor = dspy.Predict(CategoryExtractionSignature)

    def extract_category(self, prompt: str):
        # Step 1 — DSPy prediction
        raw_cat = ""
        try:
            result = self.predictor(prompt=prompt)
            raw_cat = result.category.strip().lower()
        except:
            raw_cat = ""

        # Step 2 — Match prompt text directly to taxonomy
        prompt_low = prompt.lower()

        for main_cat, keywords in CONTROLLED_CATEGORIES.items():
            for kw in keywords:
                if kw.lower() in prompt_low:
                    return main_cat

        # Step 3 — Try matching DSPy output
        for main_cat, keywords in CONTROLLED_CATEGORIES.items():
            if main_cat.lower() in raw_cat:
                return main_cat

            for kw in keywords:
                if kw.lower() in raw_cat:
                    return main_cat

        # Step 4 — Fallback
        return "General"
