import os
import dspy
from dotenv import load_dotenv
from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent
from agent.reach_agent import ReachEstimatorAgent
from agent.brand_lift_agent import BrandLiftAgent
import json


load_dotenv()

dspy.configure(
    lm=dspy.LM(
        model = "ollama/phi3",
        # model="ollama/phi4-reasoning",
        max_tokens=4096,
        temperature=0.2,
    )
)

prompts_path = os.path.join("prompts", "sample_prompts.json")

with open(prompts_path, "r") as f:
    prompts = json.load(f)

print(f"Loaded {len(prompts)} sample prompts.")

def main():
    brand_agent = BrandAgent()
    category_agent = CategoryAgent()
    # reach_agent = ReachEstimatorAgent()
    # lift_agent = BrandLiftAgent()

    for prompt in prompts:
        brands = brand_agent.extract_brand(prompt)
        category = category_agent.extract_category(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"→ Brands: {brands}")
        print(f"→ Category: {category}")

if __name__ == "__main__":
    main()