import os
import json
import csv
from datetime import datetime
from llm_setup import invoke_bedrock_model

# Paths for prompts
json_prompts_path = os.path.join("prompts", "sample_prompts.json")
csv_prompts_path = os.path.join("prompts", "prompts.csv")

# Load prompts: prefer CSV if exists, otherwise JSON
prompts_list = []

if os.path.exists(csv_prompts_path):
    print(f"Found CSV prompts at: {csv_prompts_path} — loading CSV")
    with open(csv_prompts_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text") or row.get("prompt") or row.get("query") or ""
            pid = row.get("id") or row.get("prompt_id") or ""
            ts = row.get("timestamp") or row.get("time") or ""
            if not pid:
                pid = str(len(prompts_list) + 1)
            prompts_list.append({
                "prompt_id": pid,
                "prompt_text": text.strip(),
                "timestamp": ts
            })

elif os.path.exists(json_prompts_path):
    print(f"CSV not found. Loading JSON prompts at: {json_prompts_path}")
    with open(json_prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data, start=1):
                if isinstance(item, str):
                    prompts_list.append({
                        "prompt_id": str(i),
                        "prompt_text": item.strip()
                    })
                elif isinstance(item, dict):
                    text = item.get("prompt") or item.get("text") or item.get("query") or ""
                    pid = item.get("id") or str(i)
                    ts = item.get("timestamp") or item.get("time") or ""
                    prompts_list.append({
                        "prompt_id": str(pid),
                        "prompt_text": text.strip(),
                        "timestamp": ts
                    })
                else:
                    continue
        else:
            raise ValueError("JSON prompts file must contain a list.")

else:
    raise FileNotFoundError(f"Neither {csv_prompts_path} nor {json_prompts_path} were found.")

print(f"Loaded {len(prompts_list)} prompts to process.")

# ============================================================
#  write output to CSV
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"brand_results_{timestamp}.csv"


def init_csv():
    """Create CSV with header."""
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_id",
            "prompt_text",
            "extracted_brands",
            "extracted_category"
        ])


def append_row(prompt_id, prompt_text, brands, category):
    """
    Append a single result row.
    brands = list of {brand, confidence}
    """
    brand_str = " | ".join(
        [f"{b['brand']} ({b['confidence']})" for b in brands]
    )

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            prompt_id,
            prompt_text,
            brand_str,
            category
        ])


# ============================================================
#  MAIN EXECUTION
# ============================================================
def main():
    init_csv()

    for item in prompts_list:
        prompt_id = item.get("prompt_id")
        prompt_text = item.get("prompt_text", "")

        print("\n" + "=" * 60)
        print(f"Processing prompt_id={prompt_id}")
        print(f"Prompt: {prompt_text}")

        # Brand extraction using Bedrock
        try:
            brand_system_prompt = """
You are a brand extraction expert. Given a user prompt, extract all brands mentioned.
Return a JSON object with:
- "brands": list of objects with "brand" (string) and "confidence" (0-1) keys
"""
            brand_response = invoke_bedrock_model(prompt_text, brand_system_prompt)
            brands = brand_response.get("brands", [])
        except Exception as e:
            print(f"[ERROR] Brand extraction failed for prompt_id={prompt_id}: {e}")
            brands = []

        # Category extraction using Bedrock
        try:
            category_system_prompt = """
You are a category classification expert. Given a user prompt, determine the product category.
Return a JSON object with:
- "category": (string) the product category
"""
            category_response = invoke_bedrock_model(prompt_text, category_system_prompt)
            category = category_response.get("category", "General")
        except Exception as e:
            print(f"[ERROR] Category extraction failed for prompt_id={prompt_id}: {e}")
            category = "General"

        # Print debug info
        print("→ Brands:")
        for b in brands:
            print(f"   - {b.get('brand', 'Unknown')} (confidence: {b.get('confidence', 'N/A')})")

        print(f"→ Category: {category}")

        # Save row
        append_row(prompt_id, prompt_text, brands, category)

    print(f"\nCSV saved as: {csv_filename}\n")


if __name__ == "__main__":
    main()