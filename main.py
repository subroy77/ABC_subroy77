import os
import dspy
from dotenv import load_dotenv
from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent

load_dotenv()

# Configure once globally
dspy.configure(lm=dspy.LM("ollama/phi3"))


prompts = [
    "Can you compare Sony and Samsung TV and tell me which is better?",
    "Show me best running shoes from Nike or Adidas",
    "Apple vs Dell — which laptop lasts longer?",
]

def main():
    brand_agent = BrandAgent()
    category_agent = CategoryAgent()

    for prompt in prompts:
        brands = brand_agent.extract_brands(prompt)
        category = category_agent.extract_category(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"→ Brands: {brands}")
        print(f"→ Category: {category}")

if __name__ == "__main__":
    main()